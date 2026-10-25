#!/usr/bin/env python3
"""test_mission_runner.py : mission_runner.py の状態遷移の単体テスト（STEP 19-3a）。

ROS 2・Gazebo・YOLO は使わない。/cmd_vel は扱わない。子プロセス・承認・向き・プロセス確認はすべて偽物。
偽の search_loop が書く CSV と、偽の spin_and_stop.py（ハッシュ照合用の中身だけのファイル）は、
テストごとの一時ディレクトリに作り、テスト終了時に自動で削除する。
実プロセスを使うのは SIGINT の引き継ぎの確認（python -c で自分の SIGINT ハンドラーを表示するだけ）だけ。
実行: python3 -B test_mission_runner.py
"""
import csv
import hashlib
import io
import math
import os
import signal
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mission_logic as ml  # noqa: E402
import mission_runner as mr  # noqa: E402
from search_loop import RAW_FIELDS  # noqa: E402
from test_mission_logic import located, no_person, row  # noqa: E402

DRY_OK = mr.ChildResult(0, '...\n結果: DRYRUN_OK\n理由: dry-run\n')
SPIN_OK = mr.ChildResult(0, '...\n結果: DONE\n理由: OK\n')


def spin_result(code):
    names = {0: 'DONE', 1: 'ABORT', 2: 'REPORT_FAIL', 3: 'PRECHECK_ABORT', 4: 'PRECHECK_NOT_STILL', 5: 'ARG_ERROR'}
    return mr.ChildResult(code, f'結果: {names.get(code, "?")}\n')


class FakeExecutor:
    """コマンドの種類（dryrun / spin / search）ごとに、用意した結果を順に返す。
    search の結果は (終了コード, 行のリストのリスト) で、行ごとに CSV を --out-dir に書く。"""

    def __init__(self, dryrun=None, spin=None, search=None, hook=None):
        self.q = {'dryrun': list(dryrun or []), 'spin': list(spin or []), 'search': list(search or [])}
        self.calls = []
        self.hook = hook

    def __call__(self, cmd):
        kind = 'dryrun' if '--dry-run' in cmd else 'spin' if '--spin' in cmd else 'search'
        self.calls.append(kind)
        if self.hook:
            self.hook(kind)
        if not self.q[kind]:
            raise AssertionError(f'想定外の {kind} 呼び出し: {self.calls}')
        res = self.q[kind].pop(0)
        if kind != 'search':
            return res
        code, csvs = res
        out_dir = cmd[cmd.index('--out-dir') + 1]
        for i, rows in enumerate(csvs):
            with open(os.path.join(out_dir, f'search_2026102{i}_000000.csv'), 'x', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
                w.writeheader()
                w.writerows(rows)
            with open(os.path.join(out_dir, f'search_2026102{i}_000000_clusters.csv'), 'x', encoding='utf-8') as f:
                f.write('cluster_id\n')
        return mr.ChildResult(code, f'結果: {"COMPLETED" if code == 0 else code}\n')

    def count(self, kind):
        return self.calls.count(kind)


class FakeApprover:
    def __init__(self, answers):
        self.answers = list(answers)
        self.timeouts = []

    def __call__(self, prompt, timeout):
        self.timeouts.append(timeout)
        return self.answers.pop(0)


def headings(values):
    vals = list(values)

    def get(round_no):
        v = vals[round_no - 1]
        if isinstance(v, Exception):
            raise v
        return v
    return get


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.scripts = os.path.join(self.root, 'scripts')
        os.makedirs(self.scripts)
        with open(os.path.join(self.scripts, 'spin_and_stop.py'), 'w', encoding='utf-8') as f:
            f.write('# fake\n')
        with open(os.path.join(self.scripts, 'spin_and_stop.py'), 'rb') as f:
            self.fake_sha = hashlib.sha256(f.read()).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, ex, heading=None, approver=None, allow_spin=True, process_check=lambda: [], tag='t', **kw):
        r = mr.MissionRunner(os.path.join(self.root, 'runs'), self.scripts, executor=ex,
                             approver=approver or FakeApprover([]), heading=heading or headings([0.0] * 30),
                             process_check=process_check, python='python3', tag=tag, allow_spin=allow_spin,
                             spin_sha256=kw.pop('sha', self.fake_sha), out=io.StringIO(), **kw)
        return r


def np(psi=0.0):
    return no_person(3, psi=psi)


class TestNormalFlows(RunnerCase):
    def test_visible_start_reports_without_spin(self):
        ex = FakeExecutor(dryrun=[DRY_OK], search=[(0, [located(3)]), (0, [located(3)])])
        r = self.runner(ex)
        self.assertEqual(r.run(), 'END_REPORTED')
        self.assertEqual(ex.count('spin'), 0)
        self.assertEqual([x.after_spin for x in r.rounds], [False, False])
        self.assertEqual([x.order for x in r.rounds], [1, 2])
        self.assertIn('救助完了・発見の保証ではない', mr.END_TEXT['END_REPORTED'])

    def test_spin_then_detect(self):
        ex = FakeExecutor(dryrun=[DRY_OK, DRY_OK], spin=[SPIN_OK],
                          search=[(10, [np(0.0)]), (0, [located(3, psi=0.92)]), (0, [located(3, psi=0.92)])])
        ap = FakeApprover([True])
        r = self.runner(ex, heading=headings([0.0, 0.92, 0.92]), approver=ap)
        self.assertEqual(r.run(), 'END_REPORTED')
        self.assertEqual(ap.timeouts, [60.0])
        self.assertEqual([x.after_spin for x in r.rounds], [False, True, False])
        self.assertEqual(r.spins, 1)

    def test_outputs_are_new_files(self):
        ex = FakeExecutor(dryrun=[DRY_OK], search=[(0, [located(3)]), (0, [located(3)])])
        r = self.runner(ex)
        r.run()
        files = sorted(os.listdir(r.mission_dir))
        self.assertEqual(files, ['mission_log.csv', 'r01', 'r02', 'report_01.json', 'report_02.json'])


class TestAbortBeforeSpin(RunnerCase):
    def test_dryrun_failures(self):
        cases = [(mr.ChildResult(3, '結果: PRECHECK_ABORT\n'), 'END_ABORT'),
                 (mr.ChildResult(0, '結果: 何か\n'), 'END_ABORT'),                 # 0 でも DRYRUN_OK がない
                 (mr.ChildResult(4, '結果: PRECHECK_NOT_STILL\n'), 'END_EMERGENCY'),
                 (mr.ChildResult(None, ''), 'END_EMERGENCY'),
                 (mr.ChildResult(None, '起動できない', started=False), 'END_ABORT')]
        for i, (res, end) in enumerate(cases):
            with self.subTest(res=res):
                ex = FakeExecutor(dryrun=[res])
                self.assertEqual(self.runner(ex, tag=f'd{i}').run(), end)
                self.assertEqual(ex.count('search') + ex.count('spin'), 0)

    def test_search_abnormal_exit(self):
        for i, code in enumerate((2, 5, 6, 8, 9, 12, 15, None)):
            with self.subTest(code=code):
                ex = FakeExecutor(dryrun=[DRY_OK], search=[(code, [np()])])
                self.assertEqual(self.runner(ex, tag=f's{i}').run(), 'END_ABORT')
                self.assertEqual(ex.count('spin'), 0)

    def test_csv_count_mismatch(self):
        for i, csvs in enumerate(([], [np(), np()])):
            with self.subTest(n=len(csvs)):
                ex = FakeExecutor(dryrun=[DRY_OK], search=[(10, csvs)])
                r = self.runner(ex, tag=f'c{i}')
                self.assertEqual(r.run(), 'END_ABORT')
                self.assertIn('CSV が', r.end_detail)
                self.assertEqual(ex.count('spin'), 0)

    def test_heading_failure_is_undetermined(self):
        ex = FakeExecutor(dryrun=[DRY_OK], search=[(10, [np()])])
        r = self.runner(ex, heading=headings([mr.HeadingUnavailable('no odom')]))
        self.assertEqual(r.run(), 'END_UNDETERMINED')
        self.assertEqual(ex.count('spin'), 0)

    def test_default_heading_is_unavailable(self):
        ex = FakeExecutor(dryrun=[DRY_OK], search=[(10, [np()])])
        r = self.runner(ex, heading=mr.odom_snapshot_unavailable)
        self.assertEqual(r.run(), 'END_UNDETERMINED')

    def test_heading_nan_or_out_of_range(self):
        for i, v in enumerate((float('nan'), 4.0)):
            ex = FakeExecutor(dryrun=[DRY_OK], search=[(10, [np()])])
            self.assertEqual(self.runner(ex, heading=headings([v]), tag=f'h{i}').run(), 'END_UNDETERMINED')

    def test_heading_mismatch_with_csv_aborts(self):
        ex = FakeExecutor(dryrun=[DRY_OK], search=[(10, [np(0.0)])])
        self.assertEqual(self.runner(ex, heading=headings([0.5])).run(), 'END_ABORT')

    def test_hash_mismatch_and_leftover_process(self):
        ex = FakeExecutor()
        self.assertEqual(self.runner(ex, sha='0' * 64, tag='a').run(), 'END_ABORT')
        self.assertEqual(self.runner(ex, process_check=lambda: ['123 python3 spin_and_stop.py'], tag='b').run(),
                         'END_ABORT')

        def broken():
            raise RuntimeError('ps なし')
        self.assertEqual(self.runner(ex, process_check=broken, tag='c').run(), 'END_ABORT')
        self.assertEqual(ex.calls, [])

    def test_existing_output_is_not_overwritten(self):
        r0 = self.runner(FakeExecutor(), tag='same')
        os.makedirs(r0.mission_dir)
        marker = os.path.join(r0.mission_dir, 'mission_log.csv')
        with open(marker, 'w', encoding='utf-8') as f:
            f.write('keep')
        ex = FakeExecutor()
        self.assertEqual(r0.run(), 'END_ABORT')
        with open(marker, encoding='utf-8') as f:
            self.assertEqual(f.read(), 'keep')
        self.assertEqual(ex.calls, [])

    def test_abnormal_rows_research_then_abort(self):
        bad = [row('ODOM_TIME_MISMATCH', c) for c in (1, 2, 3)]
        ex = FakeExecutor(dryrun=[DRY_OK], search=[(10, [list(bad)]), (10, [[dict(x, img_stamp=f'{i}9.0') for i, x in enumerate(bad)]]),
                                                   (10, [[dict(x, img_stamp=f'{i}8.0') for i, x in enumerate(bad)]])])
        r = self.runner(ex)
        self.assertEqual(r.run(), 'END_ABORT')
        self.assertEqual(ex.count('search'), 3)
        self.assertEqual(ex.count('spin'), 0)


class TestApprovalAndSpin(RunnerCase):
    def first_spin(self, approver, spin=None, dryrun=None, tag='t'):
        ex = FakeExecutor(dryrun=dryrun or [DRY_OK], spin=spin or [], search=[(10, [np()])])
        r = self.runner(ex, approver=approver, tag=tag)
        return r, ex, r.run()

    def test_rejected_or_timeout(self):
        for i, ans in enumerate((False, None, 'y')):          # 'y' という文字列も True ではないので拒否
            with self.subTest(ans=ans):
                r, ex, end = self.first_spin(FakeApprover([ans]), tag=f'a{i}')
                self.assertEqual(end, 'END_HELD')
                self.assertEqual(ex.count('spin'), 0)

    def test_spin_disabled_does_not_ask(self):
        ap = FakeApprover([])
        ex = FakeExecutor(dryrun=[DRY_OK], search=[(10, [np()])])
        self.assertEqual(self.runner(ex, approver=ap, allow_spin=False).run(), 'END_HELD')
        self.assertEqual(ap.timeouts, [])
        self.assertEqual(ex.count('spin'), 0)

    def test_main_keeps_spin_disabled(self):
        import inspect
        src = inspect.getsource(mr.main)
        self.assertNotIn('allow_spin=True', src)
        self.assertFalse(inspect.signature(mr.MissionRunner).parameters['allow_spin'].default)

    def test_spin_exit_codes(self):
        cases = {1: 'END_ABORT', 2: 'END_EMERGENCY', 3: 'END_ABORT', 4: 'END_EMERGENCY', 5: 'END_ABORT',
                 7: 'END_EMERGENCY', -2: 'END_EMERGENCY', None: 'END_EMERGENCY'}
        for i, (code, end) in enumerate(cases.items()):
            with self.subTest(code=code):
                r, ex, got = self.first_spin(FakeApprover([True]), spin=[spin_result(code)], tag=f'x{i}')
                self.assertEqual(got, end)
                self.assertEqual(ex.count('search'), 1)        # 旋回後の探索は自動で続けない
                self.assertEqual(ex.count('dryrun'), 1)        # 旋回後の確認にも進まない

    def test_spin_zero_without_done_line(self):
        r, ex, got = self.first_spin(FakeApprover([True]), spin=[mr.ChildResult(0, '結果: ABORT\n')])
        self.assertEqual(got, 'END_EMERGENCY')

    def test_spin_not_started(self):
        r, ex, got = self.first_spin(FakeApprover([True]), spin=[mr.ChildResult(None, 'x', started=False)])
        self.assertEqual(got, 'END_ABORT')
        self.assertEqual(r.spins, 0)

    def test_postcheck_failure(self):
        for i, res in enumerate((mr.ChildResult(3, '結果: PRECHECK_ABORT\n'), mr.ChildResult(4, '結果: PRECHECK_NOT_STILL\n'))):
            with self.subTest(res=res):
                r, ex, got = self.first_spin(FakeApprover([True]), spin=[SPIN_OK], dryrun=[DRY_OK, res], tag=f'p{i}')
                self.assertEqual(got, 'END_EMERGENCY')
                self.assertEqual(ex.count('search'), 1)


class TestFullTurn(RunnerCase):
    def sequence(self, step, n_spins_available=10):
        hs = [ml.wrap(step * i) for i in range(n_spins_available + 1)]
        ex = FakeExecutor(dryrun=[DRY_OK] * 20, spin=[SPIN_OK] * n_spins_available,
                          search=[(10, [np(h)]) for h in hs])
        r = self.runner(ex, heading=headings(hs), approver=FakeApprover([True] * 20))
        return r, ex, r.run()

    def test_seven_spins_full_turn(self):
        r, ex, end = self.sequence(0.92)
        self.assertEqual(end, 'END_SEARCHED')
        self.assertEqual(ex.count('spin'), 7)
        self.assertTrue(r.last_eval.full_turn)
        self.assertEqual(r.last_eval.decision, 'DONE_NOT_FOUND')

    def test_seven_spins_not_full_turn(self):
        r, ex, end = self.sequence(0.89)                   # 7 × 0.89 = 6.23 < 2π
        self.assertEqual(end, 'END_SEARCHED')
        self.assertEqual(ex.count('spin'), 7)               # 上限を超えて回らない
        self.assertFalse(r.last_eval.full_turn)
        self.assertEqual(r.last_eval.decision, 'DONE_SPIN_LIMIT')
        self.assertIn('一周は未確認', r.end_detail)


class TestInterrupt(RunnerCase):
    def test_interrupt_during_spin_waits_and_aborts(self):
        holder = {}

        def hook(kind):
            if kind == 'spin':
                holder['r'].request_interrupt()          # 子の実行中に Ctrl+C が来た想定
        for i, res in enumerate((SPIN_OK, spin_result(1))):
            with self.subTest(res=res):
                ex = FakeExecutor(dryrun=[DRY_OK, DRY_OK], spin=[res], search=[(10, [np()])], hook=hook)
                r = self.runner(ex, approver=FakeApprover([True]), tag=f'i{i}')
                holder['r'] = r
                self.assertEqual(r.run(), 'END_ABORT')
                self.assertEqual(ex.calls, ['dryrun', 'search', 'spin'])   # 旋回の後は何も実行しない

    def test_interrupt_during_search(self):
        holder = {}
        ex = FakeExecutor(dryrun=[DRY_OK], search=[(10, [np()])],
                          hook=lambda k: holder['r'].request_interrupt() if k == 'search' else None)
        r = self.runner(ex, approver=FakeApprover([True]))
        holder['r'] = r
        self.assertEqual(r.run(), 'END_ABORT')
        self.assertEqual(ex.count('spin'), 0)

    def test_no_terminate_or_kill_in_source(self):
        with open(mr.__file__, encoding='utf-8') as f:
            src = f.read()
        for word in ('.terminate(', '.kill(', 'os.kill', 'SIGKILL', 'signal.SIG_IGN'):
            self.assertNotIn(word, src)

    def test_child_does_not_inherit_ignored_sigint(self):
        """実プロセス：親がハンドラーを置いた状態で起動した子 Python の SIGINT が既定のハンドラーであること。"""
        flags = []
        out = io.StringIO()
        ex = mr.SubprocessExecutor(lambda: flags.append(1), out)
        code = 'import signal; print(signal.getsignal(signal.SIGINT) is signal.default_int_handler)'
        before = signal.getsignal(signal.SIGINT)
        res = ex([sys.executable, '-c', code])
        self.assertEqual(res.returncode, 0)
        self.assertEqual(res.output.strip(), 'True')
        self.assertIs(signal.getsignal(signal.SIGINT), before)   # 終了後に元へ戻す

    def test_executor_installs_flag_handler_while_waiting(self):
        flags = []
        ex = mr.SubprocessExecutor(lambda: flags.append(1), io.StringIO())
        seen = {}
        orig = mr.subprocess.Popen

        def spy(*a, **k):
            h = signal.getsignal(signal.SIGINT)
            seen['callable'] = callable(h) and h is not signal.default_int_handler
            h(signal.SIGINT, None)                       # ハンドラーを直接呼んでも例外にならず印が立つだけ
            return orig(*a, **k)
        mr.subprocess.Popen = spy
        try:
            res = ex([sys.executable, '-c', 'print(1)'])
        finally:
            mr.subprocess.Popen = orig
        self.assertTrue(seen['callable'])
        self.assertEqual(flags, [1])
        self.assertEqual(res.returncode, 0)

    def test_executor_start_failure(self):
        ex = mr.SubprocessExecutor(lambda: None, io.StringIO())
        res = ex([os.path.join(tempfile.gettempdir(), 'no_such_program_19_3a')])
        self.assertFalse(res.started)


class TestApproverInput(unittest.TestCase):
    def test_answers(self):
        cases = {'y\n': True, 'YES\n': True, 'n\n': False, '\n': False, 'yy\n': False, None: False}
        for ans, want in cases.items():
            with self.subTest(ans=ans):
                with contextlib_redirect():
                    self.assertIs(mr.stdin_approver('?', 60, read_line=lambda t: ans), want)

    def test_timeout_value_and_error(self):
        seen = []
        with contextlib_redirect():
            mr.stdin_approver('?', mr.APPROVAL_TIMEOUT_S, read_line=lambda t: seen.append(t))
        self.assertEqual(seen, [60.0])

        def boom(t):
            raise OSError('stdin なし')
        with contextlib_redirect():
            self.assertFalse(mr.stdin_approver('?', 60, read_line=boom))


def contextlib_redirect():
    import contextlib
    return contextlib.redirect_stdout(io.StringIO())


class TestResultLine(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(mr.result_line('a\n結果: DRYRUN_OK\n'), 'DRYRUN_OK')
        self.assertEqual(mr.result_line('結果: COMPLETED_NO_LOCATION\n詳細: x'), 'COMPLETED_NO_LOCATION')
        self.assertIsNone(mr.result_line('なし'))
        self.assertTrue(math.isfinite(ml.HFOV_RAD))


if __name__ == '__main__':
    unittest.main(verbosity=2)
