#!/usr/bin/env python3
"""rescue_mission.py : 探索結果を統合し、次の行動を表示する（STEP 19-2、計画モードのみ）。

しないこと：ROS 2・Gazebo の起動、/cmd_vel への送信、ロボットの移動・旋回、探索スクリプトの実行。
入力の CSV は読むだけ。--out を指定したときだけ、新しいファイルに JSON を書く（既存ファイルは上書きしない）。
判定は mission_logic.evaluate() をそのまま使う。

使い方
  python3 -B rescue_mission.py --plan \\
      --round T2 /path/search_A.csv 0.0 stay \\
      --round S1 /path/search_B.csv 0.0 stay \\
      [--expect-cumulative RAD] [--unordered] [--out report.json]

  --round ID CSV PSI SPIN（観測ラウンドごとに1回。書いた順が観測の順番）
    ID   : ラウンド名（重複不可）
    CSV  : search_loop.py の 1 周ごとの CSV（_clusters.csv ではない）
    PSI  : そのラウンドのときのロボットの向き [rad]（/odom の ψ、旋回後の値）。不明なら ?
    SPIN : 直前に旋回したなら spin、同じ向きでの再観測なら stay。不明なら ?
  --unordered          : 書いた順を観測の順番として扱わない（一周・次の行動は判定しない）
  --expect-cumulative  : 累積の回転角 [rad] の期待値。計算値と 0.05 rad を超えて違えば ABORT

終了コード: 0=判定を表示（DONE_REPORTED / RESEARCH / SPIN_NEXT / DONE_NOT_FOUND / DONE_SPIN_LIMIT）
            3=UNDETERMINED  4=ABORT  2=引数の誤り  12=WRITE_ERROR
"""
import argparse
import json
import math
import os
import sys

import mission_logic as ml

EXIT = {'DONE_REPORTED': 0, 'RESEARCH': 0, 'SPIN_NEXT': 0, 'DONE_NOT_FOUND': 0, 'DONE_SPIN_LIMIT': 0,
        'UNDETERMINED': 3, 'ABORT': 4, 'WRITE_ERROR': 12}
CUMULATIVE_TOL_RAD = 0.05     # --expect-cumulative と計算値の差の許容（暫定）
UNKNOWN = '?'
SPIN_WORDS = {'spin': True, 'stay': False}

MEANINGS = [
    'REPORTED: 同じ空間クラスタを別の観測ラウンドでも再観測し、設定した条件（根拠 {n} 回以上・{r} ラウンド以上・'
    '座標ごとの標準偏差 {s} m 以下・信頼度 {c} 以上・端の印なし）を満たしたという意味。'
    'シミュレーション上の判定で、実際の要救助者の発見や位置の正しさを保証しない',
    'NO_PERSON: その周で人物が検出されなかったという記録。その向きに人物がいないことは意味しない',
    '異常状態（ODOM_TIME_MISMATCH・CYCLE_ERROR・NO_IMAGE・NO_ODOM）は正常な未検出として扱わない',
]

NEXT_ACTION = {
    'DONE_REPORTED': '探索を終え、REPORTED のクラスタを報告する（上の意味の範囲で）',
    'RESEARCH': '旋回せず、同じ向きで search_loop.py をもう一度実行する',
    'SPIN_NEXT': 'spin_and_stop.py --spin → --dry-run → search_loop.py の順に1回ずつ実行する',
    'DONE_NOT_FOUND': 'この場所での探索を終える（一周観測したが確認条件を満たさない。不在の証明ではない）',
    'DONE_SPIN_LIMIT': 'この場所での探索を終える（旋回の上限。一周は成立していない）',
    'UNDETERMINED': '判定に必要な情報（順番・向き・旋回の有無）を補ってから、もう一度計画モードを実行する',
    'ABORT': '何も実行しない。入力と状態を確認する',
}


def parse_args(argv):
    p = argparse.ArgumentParser(description='探索結果の統合と次の行動の表示（計画モードのみ）')
    p.add_argument('--plan', action='store_true', required=True, help='計画モード（必須。ほかのモードはない）')
    p.add_argument('--round', nargs=4, action='append', metavar=('ID', 'CSV', 'PSI', 'SPIN'), required=True)
    p.add_argument('--unordered', action='store_true')
    p.add_argument('--expect-cumulative', type=float, default=None, metavar='RAD')
    p.add_argument('--out', default=None, help='JSON の出力先（新規ファイルだけ）')
    args = p.parse_args(argv)
    for rid, _csv, psi, spin in args.round:
        if psi != UNKNOWN:
            try:
                v = float(psi)
            except ValueError:
                p.error(f'--round {rid}: PSI が数値ではない: {psi!r}')
            if not math.isfinite(v) or abs(v) > math.pi + 1e-6:
                p.error(f'--round {rid}: PSI は -π〜π の rad で指定すること: {psi!r}')
        if spin != UNKNOWN and spin not in SPIN_WORDS:
            p.error(f'--round {rid}: SPIN は spin / stay / ? のどれか: {spin!r}')
    if args.expect_cumulative is not None and not math.isfinite(args.expect_cumulative):
        p.error('--expect-cumulative が有限の数ではない')
    return args


def build_rounds(args, loader=ml.load_search_csv):
    """(RoundInput のリスト, 読み込みエラー) を返す。loader はテストで差し替えられる。"""
    rounds, errors = [], []
    for i, (rid, path, psi, spin) in enumerate(args.round):
        src = os.path.normcase(os.path.abspath(path))
        try:
            rows = loader(path)
        except (ml.InputError, OSError, UnicodeDecodeError) as e:
            errors.append(f'{rid}: CSV を読めない: {type(e).__name__}: {e}')
            rows = []
        rounds.append(ml.RoundInput(
            round_id=rid, rows=rows, source=src,
            order=None if args.unordered else i + 1,
            psi_rad=None if psi == UNKNOWN else float(psi),
            after_spin=None if spin == UNKNOWN else SPIN_WORDS[spin]))
    return rounds, errors


def plan(args, loader=ml.load_search_csv):
    """Evaluation を返す。読み込みエラーや累積角の不一致は安全側（ABORT / UNDETERMINED）に倒す。"""
    rounds, errors = build_rounds(args, loader)
    if errors:
        return ml.Evaluation(decision='ABORT', reasons=['入力の CSV を読めないため、報告も終了判定もしない'],
                             errors=errors)
    ev = ml.evaluate(rounds)
    if args.expect_cumulative is not None and ev.decision != 'ABORT':
        if ev.cumulative_rad is None:
            ev.reasons.append('累積角の期待値が指定されたが、順番・向きの情報が不足していて照合できない')
            if ev.decision != 'DONE_REPORTED':
                ev.decision = 'UNDETERMINED'
        elif abs(ev.cumulative_rad - args.expect_cumulative) > CUMULATIVE_TOL_RAD:
            ev.errors.append(f'累積角の計算値 {ev.cumulative_rad:.4f} rad と期待値 {args.expect_cumulative:.4f} rad が '
                             f'{CUMULATIVE_TOL_RAD} rad を超えて違う')
            ev.decision = 'ABORT'
            ev.reasons = ['累積角の不一致のため、報告も終了判定もしない']
    return ev


def _f(v, nd=4):
    return '-' if v is None else f'{v:.{nd}f}'


def report_dict(ev, args):
    cl = []
    for c in ev.clusters:
        mean = c.mean() if c.evidence else None
        cl.append({'cluster_id': c.cluster_id, 'state': c.state, 'n_evidence': len(c.evidence),
                   'n_records': len(c.records), 'evidence_rounds': c.evidence_rounds(),
                   'mean_XY': mean, 'std_XY': c.std(), 'notes': c.notes,
                   'unverified_distance': any(o.unverified_distance for o in c.evidence + c.records)})
    return {
        'mode': 'plan', 'decision': ev.decision, 'next_action': NEXT_ACTION.get(ev.decision, ''),
        'inputs': [{'order': None if args.unordered else i + 1, 'round_id': r[0], 'csv': r[1],
                    'psi_rad': r[2], 'spin': r[3]} for i, r in enumerate(args.round)],
        'n_rounds': len(args.round), 'rounds': [vars(s) for s in ev.rounds], 'clusters': cl,
        'spins': ev.spins, 'cumulative_rad': ev.cumulative_rad, 'full_turn': ev.full_turn,
        'full_turn_judgeable': ev.cumulative_rad is not None, 'expect_cumulative': args.expect_cumulative,
        'reasons': ev.reasons, 'errors': ev.errors, 'notes': ev.notes, 'meanings': meanings(),
    }


def meanings():
    return [m.format(n=ml.MIN_EVIDENCE, r=ml.MIN_EVIDENCE_ROUNDS, s=ml.MAX_STD, c=ml.MIN_CONF) for m in MEANINGS]


def format_report(ev, args):
    L = ['=' * 72, '計画モード（ロボットは動かさない。/cmd_vel には送らない。探索も実行しない）', '=' * 72]
    L.append(f'入力（観測ラウンド {len(args.round)} 個、' + ('順番は指定なし' if args.unordered else '書いた順が観測の順番') + '）')
    for i, (rid, path, psi, spin) in enumerate(args.round):
        L.append(f'  {"-" if args.unordered else i + 1:>2}. {rid:<8} psi={psi:<9} spin={spin:<5} {path}')
    if ev.rounds:
        L.append('観測ラウンド（状態ごとの周の数。observed=その向きを観測済みに数えるか）')
        for s in ev.rounds:
            counts = ', '.join(f'{k}={v}' for k, v in sorted(s.counts.items())) or '行なし'
            L.append(f'  {s.round_id:<8} observed={"yes" if s.observed else "no ":<3} '
                     f'累積={_f(s.unwrapped_rad, 3)} rad  {counts}')
    L.append(f'クラスタ（半径 {ml.MERGE_RADIUS} m の暫定。同じ人物であることは保証しない）: {len(ev.clusters)} 個')
    for c in ev.clusters:
        mean, std = (c.mean() if c.evidence else None), c.std()
        L.append(f'  [{c.cluster_id}] {c.state:<9} 根拠 {len(c.evidence)} 件 / 記録のみ {len(c.records)} 件 / '
                 f'ラウンド {c.evidence_rounds()}')
        L.append(f'      平均 ({_f(mean and mean[0])}, {_f(mean and mean[1])})  '
                 f'標準偏差 ({_f(std and std[0])}, {_f(std and std[1])})')
        for n in c.notes:
            L.append(f'      注: {n}')
    if ev.cumulative_rad is None:
        L.append('一周の判定: 判定できない（順番・向き・旋回の有無の情報が不足、または入力が不正）')
    else:
        L.append(f'一周の判定: {"成立" if ev.full_turn else "未成立"}（旋回 {ev.spins} 回、累積 {ev.cumulative_rad:.4f} rad、'
                 f'2π = {ml.FULL_TURN_RAD:.4f}）')
    if args.expect_cumulative is not None:
        L.append(f'  累積角の期待値: {args.expect_cumulative:.4f} rad（許容 ±{CUMULATIVE_TOL_RAD}）')
    for e in ev.errors:
        L.append(f'エラー: {e}')
    for r in ev.reasons:
        L.append(f'理由: {r}')
    L.append('-' * 72)
    L.append(f'判定: {ev.decision}')
    L.append(f'次の行動: {NEXT_ACTION.get(ev.decision, "")}')
    L.append('（計画モードは表示するだけ。実行には別途承認が必要）')
    L.append('-' * 72)
    for m in meanings():
        L.append(f'※ {m}')
    for n in ev.notes:
        if n.startswith(('NO_PERSON', 'REPORTED')):   # 上の meanings() で詳しく説明済み
            continue
        L.append(f'※ {n}')
    return '\n'.join(L)


def write_json(path, data, args):
    """新規ファイルだけに書く。既存ファイル・入力 CSV と同じパスは拒否する。"""
    target = os.path.normcase(os.path.abspath(path))
    if any(target == os.path.normcase(os.path.abspath(r[1])) for r in args.round):
        raise FileExistsError(f'入力の CSV と同じパス: {path}')
    with open(path, 'x', encoding='utf-8') as f:   # 'x' = 既存ファイルがあれば失敗する
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(argv=None, loader=ml.load_search_csv, out=sys.stdout):
    args = parse_args(argv)
    ev = plan(args, loader)
    print(format_report(ev, args), file=out)
    if args.out:
        try:
            write_json(args.out, report_dict(ev, args), args)
            print(f'記録: {args.out}', file=out)
        except Exception as e:  # noqa: BLE001
            print(f'結果: WRITE_ERROR（{type(e).__name__}: {e}）', file=out)
            return EXIT['WRITE_ERROR']
    return EXIT[ev.decision]


if __name__ == '__main__':
    sys.exit(main())
