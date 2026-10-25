#!/usr/bin/env python3
"""test_mission_runner_heading.py : mission_runner.py と odom_snapshot.py の接続（STEP 19-3c）の単体テスト。

ROS 2・Gazebo・YOLO は使わない。/cmd_vel は扱わない。子プロセスはすべて偽物（odom_snapshot.py の出力も偽物）。
偽の spin_and_stop.py / odom_snapshot.py（ハッシュ照合用の中身だけ）と偽の探索 CSV は一時ディレクトリに作り、
テスト終了時に自動で削除する。既存のテストファイルは変更せず、部品を import して使う。
実行: python3 -B test_mission_runner_heading.py
"""
import hashlib
import inspect
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mission_runner as mr  # noqa: E402
import odom_snapshot as osn  # noqa: E402
from test_mission_logic import located, no_person  # noqa: E402
from test_mission_runner import DRY_OK, FakeExecutor, RunnerCase  # noqa: E402
from test_odom_snapshot import good_output  # noqa: E402


class OdomAwareExecutor(FakeExecutor):
    """odom_snapshot.py の呼び出しだけ別の結果の列から返し、それ以外は FakeExecutor に任せる。"""

    def __init__(self, odom=None, odom_hook=None, **kw):
        super().__init__(**kw)
        self.odom_q = list(odom or [])
        self.odom_cmds = []
        self.odom_hook = odom_hook

    def __call__(self, cmd):
        if cmd and cmd[-1].endswith('odom_snapshot.py'):
            self.calls.append('odom')
            self.odom_cmds.append(cmd)
            if self.odom_hook:
                self.odom_hook()
            if not self.odom_q:
                raise AssertionError(f'想定外の odom 呼び出し: {self.calls}')
            return self.odom_q.pop(0)
        return super().__call__(cmd)


def odom_ok(yaw=0.0):
    return mr.ChildResult(0, good_output(yaw))


class HeadingCase(RunnerCase):
    def setUp(self):
        super().setUp()
        self.odom_path = os.path.join(self.scripts, 'odom_snapshot.py')
        with open(self.odom_path, 'w', encoding='utf-8') as f:
            f.write('# fake odom_snapshot\n')
        with open(self.odom_path, 'rb') as f:
            self.odom_sha = hashlib.sha256(f.read()).hexdigest()

    def odom_runner(self, ex, tag='t', odom_sha=None, process_check=lambda: [], **kw):
        return mr.MissionRunner(os.path.join(self.root, 'runs'), self.scripts, executor=ex,
                                approver=kw.pop('approver', None) or (lambda p, t: False),
                                process_check=process_check, python='python3', tag=tag,
                                allow_spin=kw.pop('allow_spin', False), spin_sha256=self.fake_sha, out=io.StringIO(),
                                use_odom_snapshot=True, odom_sha256=odom_sha or self.odom_sha, **kw)


class TestNormal(HeadingCase):
    def test_yaw_from_odom_snapshot_is_used(self):
        ex = OdomAwareExecutor(odom=[odom_ok(0.0), odom_ok(0.0)], dryrun=[DRY_OK],
                               search=[(0, [located(3)]), (0, [located(3)])])
        r = self.odom_runner(ex)
        self.assertEqual(r.run(), 'END_REPORTED')
        self.assertEqual([x.psi_rad for x in r.rounds], [0.0, 0.0])
        self.assertEqual(ex.calls, ['dryrun', 'search', 'odom', 'search', 'odom'])
        self.assertEqual(ex.odom_cmds[0], ['python3', '-B', self.odom_path])

    def test_nonzero_yaw_and_spin_next_is_held(self):
        ex = OdomAwareExecutor(odom=[odom_ok(0.9224)], dryrun=[DRY_OK], search=[(10, [no_person(3, psi=0.9224)])])
        r = self.odom_runner(ex)
        self.assertEqual(r.run(), 'END_HELD')                 # allow_spin=False のため旋回しない
        self.assertAlmostEqual(r.rounds[0].psi_rad, 0.9224, places=9)
        self.assertEqual(ex.count('spin'), 0)

    def test_goes_through_executor_and_default_is_subprocess_executor(self):
        r = mr.MissionRunner(os.path.join(self.root, 'runs'), self.scripts, use_odom_snapshot=True, tag='x',
                             out=io.StringIO())
        self.assertIsInstance(r.executor, mr.SubprocessExecutor)
        seen = []
        r.executor = lambda cmd: seen.append(cmd) or odom_ok(0.5)   # run_child は self.executor を呼ぶ
        self.assertAlmostEqual(r.heading(1), 0.5, places=9)
        self.assertEqual(seen, [['python3', '-B', self.odom_path]])


class TestFailures(HeadingCase):
    def assert_undetermined(self, odom_result, tag):
        ex = OdomAwareExecutor(odom=[odom_result], dryrun=[DRY_OK], search=[(10, [no_person(3)])])
        r = self.odom_runner(ex, tag=tag)
        self.assertEqual(r.run(), 'END_UNDETERMINED')
        self.assertEqual(r.rounds, [])
        self.assertEqual(ex.count('spin'), 0)
        self.assertIn('向きを取得できない', r.end_detail)

    def test_bad_outputs(self):
        line = good_output(0.3)
        cases = {
            'exit 9 (STALE)': mr.ChildResult(9, '結果: STALE\n'),
            'exit 7 (NO_MESSAGE)': mr.ChildResult(7, '結果: NO_MESSAGE\n'),
            'not started': mr.ChildResult(None, 'x', started=False),
            'status unknown': mr.ChildResult(None, line),
            'missing line': mr.ChildResult(0, '結果: OK\n'),
            'bad json': mr.ChildResult(0, f'{osn.LINE_TAG}{{broken\n結果: OK\n'),
            'yaw mismatch': mr.ChildResult(0, good_output(0.3, yaw=0.4)),
            'stale age': mr.ChildResult(0, good_output(0.3, age=0.6)),
            'negative age': mr.ChildResult(0, good_output(0.3, age=-0.01)),
            'one sample': mr.ChildResult(0, good_output(0.3, n_samples=1)),
            'result not OK': mr.ChildResult(0, line.replace('結果: OK', '結果: STALE')),
        }
        for i, (name, res) in enumerate(cases.items()):
            with self.subTest(case=name):
                self.assert_undetermined(res, f'f{i}')

    def test_heading_mismatch_with_csv_aborts(self):
        ex = OdomAwareExecutor(odom=[odom_ok(0.5)], dryrun=[DRY_OK], search=[(10, [no_person(3, psi=0.0)])])
        self.assertEqual(self.odom_runner(ex).run(), 'END_ABORT')   # evaluate() の照合（0.02 rad）で止まる


class TestInitChecks(HeadingCase):
    def test_sha_mismatch(self):
        ex = OdomAwareExecutor()
        r = self.odom_runner(ex, odom_sha='0' * 64)
        self.assertEqual(r.run(), 'END_ABORT')
        self.assertIn('odom_snapshot.py の SHA-256', r.end_detail)
        self.assertEqual(ex.calls, [])

    def test_file_missing(self):
        os.remove(self.odom_path)
        ex = OdomAwareExecutor()
        r = self.odom_runner(ex)
        self.assertEqual(r.run(), 'END_ABORT')
        self.assertIn('odom_snapshot.py を読めない', r.end_detail)
        self.assertEqual(ex.calls, [])

    def test_expected_hash_constant(self):
        self.assertEqual(mr.ODOM_SNAPSHOT_SHA256,
                         'ee51507e788557fed052448bfb18b888122d1a7c4156f139601ca55f9b4b573d')
        real = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'odom_snapshot.py')
        self.assertEqual(mr.sha256_file(real), mr.ODOM_SNAPSHOT_SHA256)

    def test_leftover_odom_snapshot_process(self):
        self.assertIn('odom_snapshot.py', mr.WATCHED_SCRIPTS)
        ex = OdomAwareExecutor()
        r = self.odom_runner(ex, process_check=lambda: ['4321 python3 -B /home/ubuntu/odom_snapshot.py'])
        self.assertEqual(r.run(), 'END_ABORT')
        self.assertEqual(ex.calls, [])

    def test_leftover_appears_before_heading(self):
        state = {'n': 0}

        def check():
            state['n'] += 1
            return ['99 python3 odom_snapshot.py'] if state['n'] >= 4 else []   # INIT・PRECHECK・OBSERVE の後
        ex = OdomAwareExecutor(dryrun=[DRY_OK], search=[(10, [no_person(3)])])
        r = self.odom_runner(ex, process_check=check)
        self.assertEqual(r.run(), 'END_ABORT')
        self.assertNotIn('odom', ex.calls)

    def test_process_check_error(self):
        def broken():
            raise RuntimeError('ps なし')
        ex = OdomAwareExecutor()
        self.assertEqual(self.odom_runner(ex, process_check=broken).run(), 'END_ABORT')
        self.assertEqual(ex.calls, [])

    def test_heading_and_flag_together_is_error(self):
        with self.assertRaises(ValueError):
            mr.MissionRunner(self.root, self.scripts, heading=lambda n: 0.0, use_odom_snapshot=True,
                             out=io.StringIO())


class TestInterruptAndMain(HeadingCase):
    def test_interrupt_during_heading(self):
        holder = {}
        ex = OdomAwareExecutor(odom=[odom_ok(0.0)], dryrun=[DRY_OK], search=[(10, [no_person(3)])],
                               odom_hook=lambda: holder['r'].request_interrupt())
        r = self.odom_runner(ex)
        holder['r'] = r
        self.assertEqual(r.run(), 'END_ABORT')
        self.assertEqual(r.rounds, [])

    def test_interrupt_during_failed_heading(self):
        holder = {}
        ex = OdomAwareExecutor(odom=[mr.ChildResult(11, '結果: INTERRUPTED\n')], dryrun=[DRY_OK],
                               search=[(10, [no_person(3)])], odom_hook=lambda: holder['r'].request_interrupt())
        r = self.odom_runner(ex)
        holder['r'] = r
        self.assertEqual(r.run(), 'END_ABORT')

    def test_main_wiring(self):
        captured = {}

        class Spy:
            def __init__(self, *a, **k):
                captured['args'], captured['kw'] = a, k

            def run(self):
                return 'END_HELD'
        orig = mr.MissionRunner
        mr.MissionRunner = Spy
        try:
            code = mr.main(['--run', '--scripts-dir', '/s', '--out-root', '/o'])
        finally:
            mr.MissionRunner = orig
        self.assertEqual(code, mr.EXIT['END_HELD'])
        self.assertTrue(captured['kw'].get('use_odom_snapshot'))
        self.assertNotIn('allow_spin', captured['kw'])            # 初期値 False のまま
        self.assertFalse(inspect.signature(mr.MissionRunner).parameters['allow_spin'].default)
        self.assertIs(inspect.signature(mr.MissionRunner).parameters['heading'].default, mr.odom_snapshot_unavailable)


if __name__ == '__main__':
    unittest.main(verbosity=2)
