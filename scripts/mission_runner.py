#!/usr/bin/env python3
"""mission_runner.py : 探索を1ラウンドずつ進める実行制御（STEP 19-3a、ホストでの単体テストまで）。

構成
  - ROS 2 は import しない。/cmd_vel の送り側は作らない。ロボットを動かすのは spin_and_stop.py だけで、
    その開始前の確認・中断条件・STOP → VERIFY は迂回しない（子プロセスとして呼ぶだけ）。
  - 判定は mission_logic.evaluate()、記録は rescue_mission.report_dict() をそのまま使う（どちらも変更しない）。
  - ラウンドの順番・向き・旋回の有無は、この実行制御が明示的に管理して evaluate() に渡す。
  - 子プロセスの実行・承認の入力・向きの取得・残りプロセスの確認は差し替え可能（テストでは偽物を使う）。

状態遷移
  INIT → PRECHECK(--dry-run) → OBSERVE(search_loop) → HEADING → EVALUATE
    DONE_REPORTED → END_REPORTED          RESEARCH → OBSERVE（旋回なし）
    SPIN_NEXT → APPROVE → SPIN(--spin) → POSTCHECK(--dry-run) → OBSERVE（旋回あり）
    DONE_NOT_FOUND / DONE_SPIN_LIMIT → END_SEARCHED     UNDETERMINED → END_UNDETERMINED     ABORT → END_ABORT
  承認なし・時間切れ・旋回が無効 → END_HELD（旋回しない）
  停止を確認できない可能性がある終わり方 → END_EMERGENCY（以降は何もしない）

意味の範囲
  NO_PERSON : その観測で人物を検出しなかったという意味だけ。不在の結論にしない。
  REPORTED  : シミュレーション上で定義した確認条件を満たしたという意味だけ。現実の救助完了・発見の保証ではない。

19-3a で未完成として扱うもの（allow_spin の初期値は False）
  - Ctrl+C の扱い：親は SIGINT を無視（SIG_IGN）せず、印を立てるだけのハンドラーを置く。
    SIG_IGN は子に引き継がれ、Python は起動時に SIGINT が無視されていると KeyboardInterrupt を設定しないため、
    spin_and_stop.py が Ctrl+C で STOP → VERIFY に入れなくなる。Python のハンドラーは exec で引き継がれない。
    ターミナルの Ctrl+C がフォアグラウンドのプロセスグループ全体（子を含む）に届くことは、実環境で未確認。
  子プロセスを terminate / kill することはない。

向きの取得（STEP 19-3c）
  use_odom_snapshot=True のとき、odom_snapshot.make_heading_provider() を self.run_child（= SubprocessExecutor）
  経由で使い、ラウンドごとに odom_snapshot.py を子プロセスとして1回実行して yaw を得る。INIT で odom_snapshot.py の
  SHA-256 を照合する（不一致・ファイルなしは END_ABORT）。取得に失敗したら END_UNDETERMINED で、旋回へ進まない。
  main() は use_odom_snapshot=True。MissionRunner の初期値は従来どおり odom_snapshot_unavailable（常に失敗）。

STEP 19-3e の追加
  - 子プロセスの外での Ctrl+C（KeyboardInterrupt）：run() で一度だけ終わりの状態を記録してから投げ直す
    （握りつぶさない）。SPIN・POSTCHECK の途中なら END_EMERGENCY（旋回後の静止を確認できていない可能性）、
    それ以外は END_ABORT。main() は投げ直された KeyboardInterrupt を受けて 0 以外の終了コードを返す。
    子プロセスの実行中は従来どおり SubprocessExecutor が印を立てるだけで、子の終了を待つ。
  - go() は状態を先に更新してから記録する（記録の途中で中断されても同じ遷移を二重に記録しない）。
    mission_log.csv に書けなかったら log_failed を立て、main() は成功（0）の代わりに LOG_ERROR（12）を返す。
  - SHA-256 の再照合：odom_snapshot.py は向きの取得の直前、spin_and_stop.py は --spin の直前に毎回照合する。
  - 残りプロセスの検出：ps -eww -o pid=,args= の各行で、1語目が Python の実行ファイルで、オプションを除いた
    最初の引数（スクリプトのパス）のファイル名が WATCHED_SCRIPTS と完全に一致するものだけを数える。
    検出できないもの：-c・-m での起動、Python 以外からの起動、名前を変えたコピー、空白を含むパス、
    ps から見えない別のコンテナ・ホストのプロセス。
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import posixpath
import re
import select
import signal
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass

import mission_logic as ml
import odom_snapshot as osn
import rescue_mission as rm

SPIN_SHA256 = '761546798da297701bec32565d192f15db72e0683af53a6ad582e1453161206d'  # STEP 9-10 で確認した版
ODOM_SNAPSHOT_SHA256 = 'ee51507e788557fed052448bfb18b888122d1a7c4156f139601ca55f9b4b573d'  # STEP 19-3b で確認した版
APPROVAL_TIMEOUT_S = 60.0
SEARCH_ARGS = ['--cycles', '3', '--duration', '15', '--period', '1.0', '--timeout', '10']  # STEP 16・17 と同じ
WATCHED_SCRIPTS = ('spin_and_stop.py', 'search_loop.py', 'locate_once.py', 'detect_once.py', 'odom_snapshot.py')
MAX_ROUNDS = (ml.SPIN_LIMIT + 1) * ml.MAX_ROUNDS_PER_HEADING   # 念のための上限（evaluate の上限より先には来ない）

END_STATES = ('END_REPORTED', 'END_SEARCHED', 'END_UNDETERMINED', 'END_ABORT', 'END_HELD', 'END_EMERGENCY')
EXIT = {'END_REPORTED': 0, 'END_SEARCHED': 0, 'END_HELD': 20, 'END_UNDETERMINED': 3, 'END_ABORT': 4,
        'END_EMERGENCY': 2, 'LOG_ERROR': 12}
PYTHON_EXE = re.compile(r'^python(\d+(\.\d+)*)?$')    # python / python3 / python3.10 など
PY_OPTS_WITH_VALUE = 'WX'                              # -W 値 / -X 値（続けて書く -Werror も可）
EMERGENCY_TEXT = ('停止を確認できていない可能性がある。Gazebo の画面で動いていないか確認すること。'
                  '対応: (1) 速度0の手動送信（別途承認） (2) Gazebo GUI で Reset World。この実行制御はこれ以上何もしない')
END_TEXT = {
    'END_REPORTED': 'シミュレーション上の確認条件を満たしたクラスタがある（現実の救助完了・発見の保証ではない）',
    'END_SEARCHED': 'この場所での探索を終了（確認条件を満たすクラスタなし。不在の証明ではない）',
    'END_UNDETERMINED': '向き・順番などの情報が確認できず、判定できない。旋回はしていない',
    'END_ABORT': '中断（入力・状態の異常、または利用者の中断）',
    'END_HELD': '旋回しないで終了（承認なし・時間切れ・旋回が無効）',
    'END_EMERGENCY': EMERGENCY_TEXT,
}


class HeadingUnavailable(Exception):
    """向きを取得できない"""


@dataclass
class ChildResult:
    returncode: object      # int、または終了状態が不明なら None
    output: str = ''
    started: bool = True    # False = 起動できなかった（何も実行されていない）


def result_line(output):
    """spin_and_stop.py / search_loop.py が最後に表示する「結果: XXX」の XXX（無ければ None）。"""
    found = None
    for line in output.splitlines():
        if line.startswith('結果: '):
            found = line[len('結果: '):].split('（')[0].strip()
    return found


# ---------------- 差し替え可能な部品（本番用） ----------------
class SubprocessExecutor:
    """子プロセスを実行し、終わるまで待つ。terminate / kill はしない。
    待っている間だけ SIGINT に「印を立てるだけ」のハンドラーを置く（SIG_IGN にはしない。理由はモジュールの説明）。"""

    def __init__(self, on_interrupt, out=sys.stdout):
        self.on_interrupt = on_interrupt
        self.out = out

    def __call__(self, cmd):
        prev = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda s, f: self.on_interrupt())
        try:
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding='utf-8', errors='replace')
            except OSError as e:
                return ChildResult(None, f'起動できない: {type(e).__name__}: {e}', started=False)
            lines = []
            try:
                with p.stdout:
                    for line in p.stdout:
                        lines.append(line)
                        print(line, end='', file=self.out, flush=True)
                rc = p.wait()
            except Exception as e:  # noqa: BLE001  終了状態が分からない。子には何もしない
                lines.append(f'[runner] 子プロセスの待機中に {type(e).__name__}: {e}\n')
                rc = None
            return ChildResult(rc, ''.join(lines))
        finally:
            signal.signal(signal.SIGINT, prev)


def stdin_approver(prompt, timeout, read_line=None):
    """y / yes のときだけ True。それ以外・空入力・時間切れ・読めないときは False（初期値は No）。"""
    if read_line is None:
        read_line = _read_line_with_timeout
    print(f'{prompt} [y/N]（{timeout:.0f}秒で No）: ', end='', flush=True)
    try:
        ans = read_line(timeout)
    except Exception as e:  # noqa: BLE001
        print(f'\n入力を読めない（{type(e).__name__}: {e}）→ No', flush=True)
        return False
    if ans is None:
        print('\n時間切れ → No', flush=True)
        return False
    norm = unicodedata.normalize('NFKC', ans).strip().lower()   # 全角の「ｙ」なども半角として判定する
    if norm in ('y', 'yes'):
        return True
    print(f'入力 {ans!r} は y / yes ではない → No', flush=True)
    return False


def _read_line_with_timeout(timeout):
    if os.name != 'posix':          # select で標準入力を待てない環境では承認しない
        return None
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.readline() if r else None


def odom_snapshot_unavailable(round_no):
    raise HeadingUnavailable('odom_snapshot.py は未実装（STEP 19-3a）')


def script_of(args):
    """ps の args（空白区切りのコマンドライン）から、Python が実行しているスクリプトのファイル名を返す。
    1語目が Python でない、-c / -m での起動、スクリプトがない場合は None。"""
    toks = args.split()
    if not toks or not PYTHON_EXE.match(posixpath.basename(toks[0])):
        return None
    i = 1
    while i < len(toks):
        t = toks[i]
        if t == '--':
            i += 1
            break
        if t.startswith('--'):            # 長いオプション（値を取るものは想定していない）
            i += 1
            continue
        if t.startswith('-') and t != '-':
            letters, i = t[1:], i + 1
            for k, ch in enumerate(letters):
                if ch in 'cm':
                    return None
                if ch in PY_OPTS_WITH_VALUE:
                    if k == len(letters) - 1:
                        i += 1                # 値は次の語
                    break                     # 値が続けて書かれている
            continue
        break
    if i >= len(toks):
        return None
    return posixpath.basename(toks[i])


def parse_ps_output(text, me):
    """'ps -eww -o pid=,args=' の出力から、WATCHED_SCRIPTS を実行している行（自分以外）を返す。"""
    found = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if not parts[0].isdigit():
            raise ValueError(f'ps の出力を解釈できない: {line!r}')
        if parts[0] == str(me) or len(parts) < 2:
            continue
        if script_of(parts[1]) in WATCHED_SCRIPTS:
            found.append(line)
    return found


def _ps_output(run=subprocess.run, osname=None):
    if (osname or os.name) != 'posix':
        raise RuntimeError('ps で確認できない環境')
    return run(['ps', '-eww', '-o', 'pid=,args='], capture_output=True, text=True, check=True).stdout


def ps_process_check(output=_ps_output):
    """監視対象のスクリプトが動いていれば、その行のリストを返す。確認できなければ例外。"""
    return parse_ps_output(output(), os.getpid())


def sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------- 実行制御 ----------------
class MissionRunner:
    def __init__(self, out_root, scripts_dir, executor=None, approver=stdin_approver, heading=odom_snapshot_unavailable,
                 process_check=ps_process_check, python='python3', tag=None, allow_spin=False,
                 spin_sha256=SPIN_SHA256, out=sys.stdout, use_odom_snapshot=False, odom_sha256=ODOM_SNAPSHOT_SHA256):
        self.out = out
        self.mission_dir = os.path.join(out_root, f'mission_{tag or time.strftime("%Y%m%d_%H%M%S")}')
        self.scripts_dir = scripts_dir
        self.spin = os.path.join(scripts_dir, 'spin_and_stop.py')
        self.search = os.path.join(scripts_dir, 'search_loop.py')
        self.odom = os.path.join(scripts_dir, 'odom_snapshot.py')
        self.executor = executor or SubprocessExecutor(self.request_interrupt, out)
        self.use_odom_snapshot, self.odom_sha256 = use_odom_snapshot, odom_sha256
        if use_odom_snapshot:
            if heading is not odom_snapshot_unavailable:
                raise ValueError('use_odom_snapshot=True と heading は同時に指定しない')
            heading = osn.make_heading_provider(self.run_child, python, self.odom)   # 既存の executor を経由する
        self.approver, self.heading, self.process_check = approver, heading, process_check
        self.python, self.allow_spin, self.spin_sha256 = python, allow_spin, spin_sha256
        self.state = 'INIT'
        self.interrupted = False
        self.rounds = []          # mission_logic.RoundInput（順番どおり）
        self.round_meta = []      # (round_id, csv, psi, after_spin)
        self.next_after_spin = False
        self.spins = 0
        self.last_eval = None
        self.history = []         # (状態, 次の状態, 詳細)
        self.log_f = self.log_w = None
        self.log_failed = False
        self.log_error = ''
        self.end_detail = ''

    # ---- 共通 ----
    def request_interrupt(self):
        self.interrupted = True

    def say(self, msg):
        print(f'[runner] {msg}', file=self.out, flush=True)

    def go(self, nxt, detail=''):
        """状態を先に更新してから記録する。記録に失敗しても状態は戻さず、log_failed を立てる。"""
        prev = self.state
        self.state = nxt
        if nxt in END_STATES:
            self.end_detail = detail
        self.history.append((prev, nxt, detail))
        if self.log_w is not None and not self.log_failed:
            try:
                self.log_w.writerow({'seq': len(self.history), 'state': prev, 'next': nxt, 'detail': detail,
                                     'rounds': len(self.rounds), 'spins': self.spins})
                self.log_f.flush()
            except Exception as e:  # noqa: BLE001
                self.log_failed = True
                self.log_error = f'{type(e).__name__}: {e}'
                self.say(f'[WARN] mission_log.csv に書けない（以降は記録しない）: {self.log_error}')
        self.say(f'{prev} → {nxt}  {detail}')

    def verify_sha(self, path, expected, name):
        """照合に成功すれば True。不一致・ファイルなし・読めないときは END_ABORT にして False。"""
        try:
            ok = sha256_file(path) == expected
        except OSError as e:
            self.go('END_ABORT', f'{name} を読めない: {type(e).__name__}: {e}')
            return False
        if not ok:
            self.go('END_ABORT', f'{name} の SHA-256 が確認済みの版と違う')
        return ok

    def run_child(self, cmd):
        self.say('実行: ' + ' '.join(cmd))
        return self.executor(cmd)

    def checked_interrupt(self):
        if self.interrupted:
            self.go('END_ABORT', '利用者の中断（Ctrl+C）。子プロセスの終了を待ってから中断した')
            return True
        return False

    # ---- 状態 ----
    def s_init(self):
        try:
            os.makedirs(self.mission_dir, exist_ok=False)          # 既存の出力先は使わない
            self.log_f = open(os.path.join(self.mission_dir, 'mission_log.csv'), 'x', newline='', encoding='utf-8')
            self.log_w = csv.DictWriter(self.log_f, fieldnames=['seq', 'state', 'next', 'detail', 'rounds', 'spins'])
            self.log_w.writeheader()
        except OSError as e:
            return self.go('END_ABORT', f'出力先を新規作成できない: {type(e).__name__}: {e}')
        if not self.verify_sha(self.spin, self.spin_sha256, 'spin_and_stop.py'):
            return
        if self.use_odom_snapshot and not self.verify_sha(self.odom, self.odom_sha256, 'odom_snapshot.py'):
            return
        if not self.check_no_leftover():
            return
        self.go('PRECHECK', f'出力先 {self.mission_dir}')

    def check_no_leftover(self):
        try:
            left = self.process_check()
        except Exception as e:  # noqa: BLE001
            self.go('END_ABORT', f'残っているプロセスを確認できない: {type(e).__name__}: {e}')
            return False
        if left:
            self.go('END_ABORT', f'探索・旋回のプロセスが残っている: {left}')
            return False
        return True

    def dryrun(self, after_spin):
        """--dry-run の結果で次の状態を決める。after_spin=True なら失敗は END_EMERGENCY。"""
        if not self.check_no_leftover():
            return False
        r = self.run_child([self.python, '-B', self.spin, '--dry-run'])
        state = result_line(r.output)
        if r.started and r.returncode == 0 and state == 'DRYRUN_OK':
            return True
        detail = f'--dry-run が DRYRUN_OK ではない（終了コード {r.returncode}、結果 {state}、起動 {r.started}）'
        if after_spin or r.returncode == 4 or (r.started and r.returncode is None):
            self.go('END_EMERGENCY', detail)           # 旋回後の静止を確認できない／停止未確認
        else:
            self.go('END_ABORT', detail)
        return False

    def s_precheck(self):
        if self.dryrun(after_spin=False) and not self.checked_interrupt():
            self.go('OBSERVE', 'DRYRUN_OK（送信の部品は作っていない）')

    def s_observe(self):
        if len(self.rounds) >= MAX_ROUNDS:
            return self.go('END_ABORT', f'ラウンド数が上限 {MAX_ROUNDS} に達した')
        if not self.check_no_leftover():
            return
        rid = f'r{len(self.rounds) + 1:02d}'
        rdir = os.path.join(self.mission_dir, rid)
        try:
            os.makedirs(rdir, exist_ok=False)
        except OSError as e:
            return self.go('END_ABORT', f'{rdir} を新規作成できない: {e}')
        r = self.run_child([self.python, '-B', self.search] + SEARCH_ARGS + ['--out-dir', rdir])
        if self.checked_interrupt():
            return
        if not r.started or r.returncode not in (0, 10):
            return self.go('END_ABORT', f'{rid}: search_loop.py の異常終了（終了コード {r.returncode}、起動 {r.started}）')
        raw = [p for p in glob.glob(os.path.join(rdir, 'search_*.csv')) if not p.endswith('_clusters.csv')]
        if len(raw) != 1:
            return self.go('END_ABORT', f'{rid}: 1周ごとの CSV が {len(raw)} 個（1 個のはず）')
        try:
            rows = ml.load_search_csv(raw[0])
        except (ml.InputError, OSError, UnicodeDecodeError) as e:
            return self.go('END_ABORT', f'{rid}: CSV を読めない: {e}')
        self.pending = (rid, raw[0], rows)
        self.go('HEADING', f'{rid}: 終了コード {r.returncode}、{len(rows)} 行')

    def s_heading(self):
        rid, path, rows = self.pending
        if self.use_odom_snapshot:
            if not self.check_no_leftover():
                return
            if not self.verify_sha(self.odom, self.odom_sha256, 'odom_snapshot.py'):   # 向きの取得の直前に再照合
                return
        try:
            psi = float(self.heading(len(self.rounds) + 1))
        except Exception as e:  # noqa: BLE001
            if self.checked_interrupt():
                return
            return self.go('END_UNDETERMINED', f'{rid}: 向きを取得できない: {type(e).__name__}: {e}')
        if self.checked_interrupt():
            return
        if not (abs(psi) <= 3.141593):
            return self.go('END_UNDETERMINED', f'{rid}: 向きが範囲外: {psi}')
        self.rounds.append(ml.RoundInput(round_id=rid, rows=rows, order=len(self.rounds) + 1, psi_rad=psi,
                                         after_spin=self.next_after_spin, source=os.path.abspath(path)))
        self.round_meta.append((rid, path, f'{psi:.4f}', 'spin' if self.next_after_spin else 'stay'))
        self.go('EVALUATE', f'{rid}: psi={psi:.4f} rad、旋回あり={self.next_after_spin}')

    def s_evaluate(self):
        ev = ml.evaluate(self.rounds)
        self.last_eval = ev
        self.write_report(ev)
        reasons = ' / '.join(ev.reasons + ev.errors)
        d = ev.decision
        if d == 'DONE_REPORTED':
            self.go('END_REPORTED', reasons)
        elif d == 'RESEARCH':
            self.next_after_spin = False
            self.go('OBSERVE', 'RESEARCH: ' + reasons)
        elif d == 'SPIN_NEXT':
            if self.spins >= ml.SPIN_LIMIT:                     # evaluate の上限判定の二重確認
                self.go('END_ABORT', f'旋回が上限 {ml.SPIN_LIMIT} 回に達しているのに SPIN_NEXT')
            else:
                self.go('APPROVE', reasons)
        elif d == 'DONE_NOT_FOUND':
            self.go('END_SEARCHED', reasons)
        elif d == 'DONE_SPIN_LIMIT':
            self.go('END_SEARCHED', '旋回の上限に達した。一周は未確認 / ' + reasons)
        elif d == 'UNDETERMINED':
            self.go('END_UNDETERMINED', reasons)
        else:
            self.go('END_ABORT', f'{d}: {reasons}')

    def s_approve(self):
        if not self.allow_spin:
            return self.go('END_HELD', '実旋回は無効（STEP 19-3a：Ctrl+C の扱いが実環境で未確認）')
        ok = self.approver(f'次の向きへ旋回します（{self.spins + 1} 回目、spin_and_stop.py --spin）。実行しますか',
                           APPROVAL_TIMEOUT_S)
        if self.checked_interrupt():
            return
        if ok is not True:
            return self.go('END_HELD', '旋回の承認がない（No・空入力・時間切れ）')
        self.go('SPIN', '承認あり')

    def s_spin(self):
        if not self.check_no_leftover():
            return
        if not self.verify_sha(self.spin, self.spin_sha256, 'spin_and_stop.py'):        # --spin の直前に再照合
            return
        r = self.run_child([self.python, '-B', self.spin, '--spin'])
        state = result_line(r.output)
        detail = f'終了コード {r.returncode}、結果 {state}、起動 {r.started}'
        if r.started and r.returncode is not None:
            self.spins += 1                     # 起動した旋回は、結果によらず回数に数える
        if not r.started:
            return self.go('END_ABORT', '旋回を起動できない（送信なし）: ' + r.output)
        if r.returncode == 0 and state == 'DONE':
            if self.checked_interrupt():
                return
            return self.go('POSTCHECK', detail)
        if r.returncode in (1, 3, 5):           # 1=停止は確認済み、3/5=送信なし
            return self.go('END_ABORT', detail)
        self.go('END_EMERGENCY', detail)       # 2=REPORT_FAIL、4=停止未確認、想定外、不明、0 なのに DONE がない

    def s_postcheck(self):
        if self.dryrun(after_spin=True) and not self.checked_interrupt():
            self.next_after_spin = True
            self.go('OBSERVE', '旋回後の静止を確認（DRYRUN_OK）')

    def write_report(self, ev):
        args = argparse.Namespace(round=self.round_meta, unordered=False, expect_cumulative=None)
        path = os.path.join(self.mission_dir, f'report_{len(self.rounds):02d}.json')
        try:
            with open(path, 'x', encoding='utf-8') as f:
                json.dump(rm.report_dict(ev, args), f, ensure_ascii=False, indent=2)
        except OSError as e:
            self.say(f'記録を書けない（判定は続ける）: {e}')
        print(rm.format_report(ev, args), file=self.out)

    # ---- 実行 ----
    def run(self):
        steps = {'INIT': self.s_init, 'PRECHECK': self.s_precheck, 'OBSERVE': self.s_observe,
                 'HEADING': self.s_heading, 'EVALUATE': self.s_evaluate, 'APPROVE': self.s_approve,
                 'SPIN': self.s_spin, 'POSTCHECK': self.s_postcheck}
        try:
            try:
                while self.state not in END_STATES:
                    steps[self.state]()
            except KeyboardInterrupt:
                # 子プロセスの外での Ctrl+C。まだ終わりの状態でなければ一度だけ記録し、投げ直す（握りつぶさない）
                self.interrupted = True
                if self.state not in END_STATES:
                    end = 'END_EMERGENCY' if self.state in ('SPIN', 'POSTCHECK') else 'END_ABORT'
                    self.go(end, f'利用者の中断（Ctrl+C、子プロセスの外、{self.state} の途中）')
                raise
        finally:
            try:
                self.say(f'終了: {self.state} — {END_TEXT.get(self.state, "")}')
            finally:
                if self.log_f is not None:
                    try:
                        self.log_f.close()
                    except Exception as e:  # noqa: BLE001
                        self.log_failed = True
                        self.log_error = f'{type(e).__name__}: {e}'
        return self.state


def main(argv=None):
    p = argparse.ArgumentParser(description='探索の実行制御（向きは odom_snapshot.py から取得。旋回は --allow-spin のときだけ）')
    p.add_argument('--run', action='store_true', required=True)
    p.add_argument('--scripts-dir', default='/home/ubuntu')
    p.add_argument('--out-root', default='/home/ubuntu/runs')
    p.add_argument('--python', default='python3')
    p.add_argument('--allow-spin', action='store_true',
                   help='旋回を有効にする（旋回のたびに端末で y の承認が必要。指定しなければ旋回しない）')
    args = p.parse_args(argv)
    extra = {}
    if args.allow_spin:                     # 指定したときだけ渡す。省略時は MissionRunner の初期値（False）のまま
        extra['allow_spin'] = True
        print('[runner] --allow-spin：旋回は有効。旋回のたびに y の承認を求める（No・空入力・60秒の時間切れは旋回しない）',
              flush=True)
    runner = MissionRunner(args.out_root, args.scripts_dir, python=args.python,
                           use_odom_snapshot=True, **extra)
    return exit_code(runner)


def exit_code(runner):
    """run() の結果を終了コードにする。Ctrl+C は 0 以外、記録の失敗は成功（0）にしない。"""
    try:
        state = runner.run()
    except KeyboardInterrupt:
        state = 'END_EMERGENCY' if getattr(runner, 'state', None) == 'END_EMERGENCY' else 'END_ABORT'
        print(f'[runner] Ctrl+C で中断: {state}', flush=True)
    code = EXIT[state]
    if getattr(runner, 'log_failed', False) and code == 0:
        print(f'[runner] mission_log.csv の記録に失敗したため成功として扱わない: {getattr(runner, "log_error", "")}',
              flush=True)
        code = EXIT['LOG_ERROR']
    return code


if __name__ == '__main__':
    sys.exit(main())
