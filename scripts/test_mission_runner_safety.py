#!/usr/bin/env python3
"""test_mission_runner_safety.py : STEP 19-3e（Ctrl+C の終了処理・SHA-256 の再照合・プロセス検出）の単体テスト。

ROS 2・Gazebo・YOLO は使わない。/cmd_vel は扱わない。子プロセスはすべて偽物。
Linux の ps は実行せず、ps の出力（文字列）と subprocess.run の偽物で確かめる。
偽のスクリプトと CSV は一時ディレクトリに作り、テスト終了時に自動で削除する。既存のテストファイルは変更しない。
実行: python3 -B test_mission_runner_safety.py
"""
import csv
import io
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mission_runner as mr  # noqa: E402
from test_mission_logic import located, no_person  # noqa: E402
from test_mission_runner import DRY_OK, SPIN_OK, FakeApprover  # noqa: E402
from test_mission_runner_heading import HeadingCase, OdomAwareExecutor, odom_ok  # noqa: E402


def read_log(r):
    with open(os.path.join(r.mission_dir, 'mission_log.csv'), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


class InterruptingStream(io.StringIO):
    """指定した文字列を含む書き込みのときに KeyboardInterrupt を出す（表示の途中での Ctrl+C の代わり）。"""

    def __init__(self, trigger):
        super().__init__()
        self.trigger, self.fired = trigger, False

    def write(self, s):
        if not self.fired and self.trigger in s:
            self.fired = True
            raise KeyboardInterrupt
        return super().write(s)


class TestKeyboardInterruptOutsideChild(HeadingCase):
    def test_during_approval_wait(self):
        def approver(prompt, timeout):
            raise KeyboardInterrupt
        ex = OdomAwareExecutor(odom=[odom_ok(0.0)], dryrun=[DRY_OK], search=[(10, [no_person(3)])])
        r = self.odom_runner(ex, approver=approver, allow_spin=True)
        with self.assertRaises(KeyboardInterrupt):           # 握りつぶさずに投げ直す
            r.run()
        self.assertEqual(r.state, 'END_ABORT')
        self.assertEqual(r.history[-1][:2], ('APPROVE', 'END_ABORT'))
        self.assertEqual(ex.count('spin'), 0)
        rows = read_log(r)
        self.assertEqual([x['next'] for x in rows].count('END_ABORT'), 1)   # 二重に記録しない
        self.assertEqual(rows[-1]['state'], 'APPROVE')

    def test_during_heading_function(self):
        def heading(n):
            raise KeyboardInterrupt
        ex = OdomAwareExecutor(dryrun=[DRY_OK], search=[(10, [no_person(3)])])
        r = mr.MissionRunner(os.path.join(self.root, 'runs'), self.scripts, executor=ex, heading=heading,
                             process_check=lambda: [], tag='h', spin_sha256=self.fake_sha, out=io.StringIO())
        with self.assertRaises(KeyboardInterrupt):
            r.run()
        self.assertEqual(r.state, 'END_ABORT')
        self.assertEqual(r.rounds, [])

    def test_during_postcheck_is_emergency(self):
        calls = {'n': 0}

        def check():
            calls['n'] += 1
            if calls['n'] == 5:            # INIT・PRECHECK・OBSERVE・SPIN の次 = POSTCHECK の確認
                raise KeyboardInterrupt
            return []
        ex = OdomAwareExecutor(dryrun=[DRY_OK], spin=[SPIN_OK], search=[(10, [no_person(3)])])
        r = mr.MissionRunner(os.path.join(self.root, 'runs'), self.scripts, executor=ex,
                             heading=lambda n: 0.0, approver=FakeApprover([True]), process_check=check, tag='p',
                             allow_spin=True, spin_sha256=self.fake_sha, out=io.StringIO())
        with self.assertRaises(KeyboardInterrupt):
            r.run()
        self.assertEqual(r.history[-1][:2], ('POSTCHECK', 'END_EMERGENCY'))
        self.assertEqual(ex.calls, ['dryrun', 'search', 'spin'])

    def test_interrupt_while_printing_end_transition_is_not_recorded_twice(self):
        ex = OdomAwareExecutor(odom=[odom_ok(0.0), odom_ok(0.0)], dryrun=[DRY_OK],
                               search=[(0, [located(3)]), (0, [located(3)])])
        r = self.odom_runner(ex)
        r.out = InterruptingStream('→ END_REPORTED')
        with self.assertRaises(KeyboardInterrupt):
            r.run()
        self.assertEqual(r.state, 'END_REPORTED')               # 状態は先に更新済み。END_ABORT を重ねない
        self.assertEqual([h[1] for h in r.history].count('END_REPORTED'), 1)
        self.assertNotIn('END_ABORT', [h[1] for h in r.history])
        self.assertEqual(read_log(r)[-1]['next'], 'END_REPORTED')

    def test_exit_code_on_interrupt_is_not_success(self):
        class R:
            state, log_failed, log_error = 'END_REPORTED', False, ''

            def run(self):
                raise KeyboardInterrupt
        with contextlib_quiet():
            self.assertEqual(mr.exit_code(R()), mr.EXIT['END_ABORT'])

        class E(R):
            state = 'END_EMERGENCY'
        with contextlib_quiet():
            self.assertEqual(mr.exit_code(E()), mr.EXIT['END_EMERGENCY'])

    def test_child_interrupt_path_unchanged(self):
        holder = {}
        ex = OdomAwareExecutor(odom=[odom_ok(0.0)], dryrun=[DRY_OK], search=[(10, [no_person(3)])])
        ex.hook = lambda kind: holder['r'].request_interrupt() if kind == 'search' else None
        r = self.odom_runner(ex)
        holder['r'] = r
        self.assertEqual(r.run(), 'END_ABORT')                   # 例外ではなく通常の終了（従来どおり）
        self.assertNotIn('odom', ex.calls)


class TestLogFailure(HeadingCase):
    def test_log_write_failure_is_not_success(self):
        holder = {}

        def break_log(kind):
            if kind == 'search':
                holder['r'].log_f.close()                       # 以降の書き込みが失敗する
        ex = OdomAwareExecutor(odom=[odom_ok(0.0), odom_ok(0.0)], dryrun=[DRY_OK],
                               search=[(0, [located(3)]), (0, [located(3)])])
        ex.hook = break_log
        r = self.odom_runner(ex)
        holder['r'] = r
        self.assertEqual(r.run(), 'END_REPORTED')
        self.assertTrue(r.log_failed)

        class Done:
            state, log_failed, log_error = 'END_REPORTED', True, 'x'

            def run(self):
                return 'END_REPORTED'
        with contextlib_quiet():
            self.assertEqual(mr.exit_code(Done()), mr.EXIT['LOG_ERROR'])

    def test_log_failure_does_not_change_failure_codes(self):
        class Abort:
            state, log_failed, log_error = 'END_ABORT', True, 'x'

            def run(self):
                return 'END_ABORT'
        self.assertEqual(mr.exit_code(Abort()), mr.EXIT['END_ABORT'])


class TestShaRecheck(HeadingCase):
    def test_odom_changed_before_heading(self):
        def tamper(kind):
            if kind == 'search':
                with open(self.odom_path, 'a', encoding='utf-8') as f:
                    f.write('# changed\n')
        ex = OdomAwareExecutor(odom=[odom_ok(0.0)], dryrun=[DRY_OK], search=[(10, [no_person(3)])])
        ex.hook = tamper
        r = self.odom_runner(ex)
        self.assertEqual(r.run(), 'END_ABORT')
        self.assertIn('odom_snapshot.py の SHA-256', r.end_detail)
        self.assertNotIn('odom', ex.calls)

    def test_odom_missing_or_unreadable_before_heading(self):
        for i, how in enumerate(('remove', 'dir')):
            with self.subTest(how=how):
                path = self.odom_path

                def tamper(kind, how=how, path=path):
                    if kind == 'search':
                        os.remove(path)
                        if how == 'dir':
                            os.makedirs(path)                   # ファイルの代わりにフォルダ（読めない）
                ex = OdomAwareExecutor(odom=[odom_ok(0.0)], dryrun=[DRY_OK], search=[(10, [no_person(3)])])
                ex.hook = tamper
                r = self.odom_runner(ex, tag=f'm{i}')
                self.assertEqual(r.run(), 'END_ABORT')
                self.assertIn('odom_snapshot.py を読めない', r.end_detail)
                self.assertNotIn('odom', ex.calls)
                if how == 'dir':
                    os.rmdir(path)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('# fake odom_snapshot\n')

    def test_spin_changed_before_spin(self):
        spin_path = os.path.join(self.scripts, 'spin_and_stop.py')

        def tamper(kind):
            if kind == 'search':
                with open(spin_path, 'a', encoding='utf-8') as f:
                    f.write('# changed\n')
        ex = OdomAwareExecutor(dryrun=[DRY_OK], spin=[SPIN_OK], search=[(10, [no_person(3)])])
        ex.hook = tamper
        r = mr.MissionRunner(os.path.join(self.root, 'runs'), self.scripts, executor=ex, heading=lambda n: 0.0,
                             approver=FakeApprover([True]), process_check=lambda: [], tag='s', allow_spin=True,
                             spin_sha256=self.fake_sha, out=io.StringIO())
        self.assertEqual(r.run(), 'END_ABORT')
        self.assertIn('spin_and_stop.py の SHA-256', r.end_detail)
        self.assertEqual(ex.count('spin'), 0)
        self.assertEqual(r.spins, 0)

    def test_spin_missing_before_spin(self):
        spin_path = os.path.join(self.scripts, 'spin_and_stop.py')
        ex = OdomAwareExecutor(dryrun=[DRY_OK], spin=[SPIN_OK], search=[(10, [no_person(3)])])
        ex.hook = lambda kind: os.remove(spin_path) if kind == 'search' else None
        r = mr.MissionRunner(os.path.join(self.root, 'runs'), self.scripts, executor=ex, heading=lambda n: 0.0,
                             approver=FakeApprover([True]), process_check=lambda: [], tag='s2', allow_spin=True,
                             spin_sha256=self.fake_sha, out=io.StringIO())
        self.assertEqual(r.run(), 'END_ABORT')
        self.assertIn('spin_and_stop.py を読めない', r.end_detail)
        self.assertEqual(ex.count('spin'), 0)

    def test_main_still_has_spin_disabled(self):
        import inspect
        self.assertFalse(inspect.signature(mr.MissionRunner).parameters['allow_spin'].default)
        code = ''.join(l.split('#')[0] for l in inspect.getsource(mr.main).splitlines())   # コメントを除いた本文
        self.assertNotIn('allow_spin=', code)


class TestProcessDetection(unittest.TestCase):
    MATCH = [
        ('python3 -B /home/ubuntu/odom_snapshot.py', 'odom_snapshot.py'),
        ('/usr/bin/python3 -B /home/ubuntu/spin_and_stop.py --spin', 'spin_and_stop.py'),
        ('python3 /home/ubuntu/spin_and_stop.py --dry-run --ros-args -r __ns:=/x', 'spin_and_stop.py'),
        ('/usr/bin/python3.10 -u -W ignore search_loop.py --cycles 3', 'search_loop.py'),
        ('python3 -Werror -B ./locate_once.py', 'locate_once.py'),
        ('python -X dev detect_once.py', 'detect_once.py'),
        ('python3 -BW ignore odom_snapshot.py', 'odom_snapshot.py'),
        ('python3 -- odom_snapshot.py', 'odom_snapshot.py'),
    ]
    NO_MATCH = [
        'python3 -B test_odom_snapshot.py',
        'python3 -B /home/ubuntu/test_search_loop_dt.py',
        'grep odom_snapshot.py',
        'tail -f /home/ubuntu/odom_snapshot.py',
        'vim /home/ubuntu/spin_and_stop.py',
        'nano search_loop.py',
        'bash -c python3 -B /home/ubuntu/odom_snapshot.py',
        'sh -c cd /home/ubuntu && python3 search_loop.py',
        'python3 -c import odom_snapshot',
        'python3 -m odom_snapshot',
        'python3 -Bm pytest spin_and_stop.py',
        'python3 -B /home/ubuntu/mission_runner.py --run',
        'python3 -B',
        'python3 /home/ubuntu/odom_snapshot.py.bak',
        'pythonw odom_snapshot.py',
        'ros2 launch /home/ubuntu/models/house_with_person_v4_headless.launch.py',
    ]

    def test_matches(self):
        for args, want in self.MATCH:
            with self.subTest(args=args):
                self.assertEqual(mr.script_of(args), want)

    def test_no_false_positive(self):
        for args in self.NO_MATCH:
            with self.subTest(args=args):
                self.assertNotIn(mr.script_of(args), mr.WATCHED_SCRIPTS)

    def test_parse_ps_output(self):
        text = '\n'.join([
            '    1 /bin/bash',
            '  100 python3 -B /home/ubuntu/mission_runner.py --run',
            '  200 python3 -B /home/ubuntu/odom_snapshot.py',
            '  201 grep odom_snapshot.py',
            '  202 python3 -B test_odom_snapshot.py',
            '  300 python3 -B /home/ubuntu/search_loop.py --cycles 3',
            '  400 [kworker/0:1]',
            '  500',
            '',
        ])
        found = mr.parse_ps_output(text, me=100)
        self.assertEqual([l.split()[0] for l in found], ['200', '300'])
        self.assertEqual(mr.parse_ps_output(text, me=200)[0].split()[0], '300')   # 自分の PID は除く

    def test_parse_ps_output_garbage_raises(self):
        with self.assertRaises(ValueError):
            mr.parse_ps_output('PID COMMAND\n 1 x\n', me=1)            # ヘッダー付き（想定外の形式）

    def test_ps_command_and_failures(self):
        seen = {}

        def fake_run(cmd, **kw):
            seen['cmd'], seen['kw'] = cmd, kw
            return subprocess.CompletedProcess(cmd, 0, stdout='  9 python3 -B /h/spin_and_stop.py --spin\n')
        out = mr._ps_output(run=fake_run, osname='posix')
        self.assertEqual(seen['cmd'], ['ps', '-eww', '-o', 'pid=,args='])
        self.assertTrue(seen['kw'].get('check'))
        self.assertEqual(mr.ps_process_check(output=lambda: out), ['9 python3 -B /h/spin_and_stop.py --spin'])
        with self.assertRaises(RuntimeError):
            mr._ps_output(run=fake_run, osname='nt')

        def failing(cmd, **kw):
            raise subprocess.CalledProcessError(1, cmd)
        with self.assertRaises(subprocess.CalledProcessError):
            mr._ps_output(run=failing, osname='posix')

    def test_runner_aborts_when_ps_fails(self):
        def broken():
            raise subprocess.CalledProcessError(1, ['ps'])
        r = mr.MissionRunner('unused_root', 'unused_dir', process_check=broken, out=io.StringIO())
        self.assertFalse(r.check_no_leftover())
        self.assertEqual(r.state, 'END_ABORT')


def contextlib_quiet():
    import contextlib
    return contextlib.redirect_stdout(io.StringIO())


if __name__ == '__main__':
    unittest.main(verbosity=2)
