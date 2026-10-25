#!/usr/bin/env python3
"""test_odom_snapshot.py : odom_snapshot.py の単体テスト（ROS なし）。

rclpy は import しない（ros_collect は呼ばない）。受信は偽の spin_once・偽の時計で与える。
mission_runner との接続は、偽の子プロセス結果と偽の探索で確かめる（一時ディレクトリは終了時に削除）。
実行: python3 -B test_odom_snapshot.py
"""
import io
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mission_runner as mr  # noqa: E402
import odom_snapshot as osn  # noqa: E402
from test_mission_logic import located, no_person  # noqa: E402
from test_mission_runner import DRY_OK, FakeExecutor, RunnerCase  # noqa: E402


def quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def odom(stamp, yaw=0.0, x=-2.0, y=-0.5, **kw):
    qx, qy, qz, qw = quat(yaw)
    d = dict(stamp=stamp, x=x, y=y, z=0.0, qx=qx, qy=qy, qz=qz, qw=qw, frame_id='odom', child_frame_id='base_footprint')
    d.update(kw)
    return osn.OdomSample(**d)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def feeder(collector, clock, events):
    """events: spin_once ごとに与える [('odom', OdomSample) / ('clock', t), ...] のリスト。尽きたら何も来ない。"""
    it = iter(events)

    def spin_once(dt):
        clock.t += dt
        for kind, v in next(it, []):
            collector.add_odom(v) if kind == 'odom' else collector.add_clock(v)
    return spin_once


def collect(events, timeout=1.0):
    clock = FakeClock()
    c = osn.SnapshotCollector(now=clock)
    return osn.run_collection(c, feeder(c, clock, events), timeout=timeout, now=clock)


GOOD = [[('clock', 10.00), ('odom', odom(9.99, 0.3))], [('clock', 10.05), ('odom', odom(10.04, 0.3))]]


class TestYaw(unittest.TestCase):
    def test_values(self):
        for yaw in (0.0, 0.5, -0.5, math.pi / 2, -math.pi / 2, 3.1, -3.1, 2.7681, -2.5926):
            with self.subTest(yaw=yaw):
                self.assertAlmostEqual(osn.yaw_from_quat(*quat(yaw)), yaw, places=12)
        self.assertAlmostEqual(abs(osn.yaw_from_quat(*quat(math.pi))), math.pi, places=12)

    def test_wrap_across_pi(self):
        self.assertAlmostEqual(osn.yaw_from_quat(*quat(math.pi + 0.2)), -math.pi + 0.2, places=12)

    def test_invalid(self):
        for q in ((0, 0, 0, 0), (0, 0, 0, 1.01), (0, 0, 0, 0.998), (float('nan'), 0, 0, 1), (0, 0, float('inf'), 1),
                  (0, 0, 0, None), (0, 0, 0, '1')):
            with self.subTest(q=q):
                with self.assertRaises(osn.SnapshotError):
                    osn.yaw_from_quat(*q)

    def test_norm_tolerance_boundary(self):
        osn.yaw_from_quat(0, 0, 0, 1.0009)              # 許容内
        with self.assertRaises(osn.SnapshotError):
            osn.yaw_from_quat(0, 0, 0, 1.0011)


class TestCollection(unittest.TestCase):
    def test_ok(self):
        st = collect(GOOD)
        self.assertEqual(st[0], 'OK')
        s = st[1]
        self.assertAlmostEqual(s.yaw, 0.3, places=12)
        self.assertAlmostEqual(s.age, 0.01, places=9)
        self.assertEqual((s.x, s.y, s.n_samples, s.stamp, s.clock), (-2.0, -0.5, 2, 10.04, 10.05))

    def test_zero_yaw_is_a_valid_success(self):
        st = collect([[('clock', 1.0), ('odom', odom(0.99, 0.0))], [('clock', 1.1), ('odom', odom(1.09, 0.0))]])
        self.assertEqual(st[0], 'OK')
        self.assertEqual(st[1].yaw, 0.0)

    def test_no_message_and_insufficient(self):
        self.assertEqual(collect([])[:2], ('FAIL', 'NO_MESSAGE'))
        self.assertEqual(collect([[('clock', 1.0), ('odom', odom(0.99))], [('clock', 1.1)]])[:2], ('FAIL', 'NO_MESSAGE'))

    def test_duplicate_stamp_is_not_a_second_sample(self):
        st = collect([[('clock', 1.0), ('odom', odom(0.99))], [('clock', 1.1), ('odom', odom(0.99))]])
        self.assertEqual(st[:2], ('FAIL', 'NO_MESSAGE'))

    def test_no_clock_and_clock_not_advancing(self):
        self.assertEqual(collect([[('odom', odom(0.9)), ('odom', odom(1.0))]])[:2], ('FAIL', 'NO_CLOCK'))
        st = collect([[('clock', 1.0), ('odom', odom(0.9)), ('odom', odom(0.95))], [('clock', 1.0)], [('clock', 1.0)]])
        self.assertEqual(st[:2], ('FAIL', 'CLOCK_NOT_ADVANCING'))

    def test_invalid_values_fail_immediately(self):
        cases = [odom(1.0, x=float('nan')), odom(float('inf')), odom(1.0, qw=2.0), odom(1.0, qx=float('nan'))]
        for s in cases:
            with self.subTest(s=s):
                st = collect([[('clock', 1.0), ('odom', odom(0.9)), ('odom', s)]] + GOOD)
                self.assertEqual(st[:2], ('FAIL', 'INVALID'))
        self.assertEqual(collect([[('clock', float('nan'))]] + GOOD)[:2], ('FAIL', 'INVALID'))

    def test_backward_time_fails_immediately(self):
        st = collect([[('clock', 1.0), ('odom', odom(0.95)), ('odom', odom(0.90))]] + GOOD)
        self.assertEqual(st[:2], ('FAIL', 'TIME_INCONSISTENT'))
        st = collect([[('clock', 1.0), ('clock', 0.5)]] + GOOD)
        self.assertEqual(st[:2], ('FAIL', 'TIME_INCONSISTENT'))

    def test_future_stamp_waits_then_resolves(self):
        ev = [[('clock', 1.00), ('odom', odom(1.00)), ('odom', odom(1.05))],     # /odom が /clock より先
              [('clock', 1.02)],
              [('clock', 1.06)]]                                                # 新しい /clock で整合
        st = collect(ev)
        self.assertEqual(st[0], 'OK')
        self.assertAlmostEqual(st[1].age, 0.01, places=9)

    def test_future_stamp_times_out(self):
        st = collect([[('clock', 1.00), ('odom', odom(1.00)), ('odom', odom(1.05))], [('clock', 1.01)]])
        self.assertEqual(st[:2], ('FAIL', 'FUTURE_STAMP'))

    def test_stale(self):
        st = collect([[('clock', 1.0), ('odom', odom(0.1)), ('odom', odom(0.2))], [('clock', 1.1)]])
        self.assertEqual(st[:2], ('FAIL', 'STALE'))

    def test_stale_boundary(self):
        st = collect([[('clock', 1.0), ('odom', odom(0.25)), ('odom', odom(0.5))], [('clock', 1.0 + 0.0)], [('clock', 1.0)]])
        self.assertEqual(st[:2], ('FAIL', 'CLOCK_NOT_ADVANCING'))
        st = collect([[('clock', 0.75), ('odom', odom(0.25)), ('odom', odom(0.5))], [('clock', 1.0)]])
        self.assertEqual(st[0], 'OK')                        # age = 0.5 ちょうどは成功
        st = collect([[('clock', 0.75), ('odom', odom(0.25)), ('odom', odom(0.5))], [('clock', 1.0001)]])
        self.assertEqual(st[:2], ('FAIL', 'STALE'))

    def test_timeout_is_respected(self):
        clock = osn.SnapshotCollector()
        ticks = {'n': 0}
        fake_now = FakeClock()

        def spin_once(dt):
            fake_now.t += dt
            ticks['n'] += 1
        c = osn.SnapshotCollector(now=fake_now)
        st = osn.run_collection(c, spin_once, timeout=0.5, now=fake_now)
        self.assertEqual(st[0], 'FAIL')
        self.assertLessEqual(ticks['n'], int(0.5 / osn.SPIN_STEP_S) + 1)
        del clock


class TestMain(unittest.TestCase):
    def run_main(self, collect_fn, argv=()):
        out = io.StringIO()
        code = osn.main(list(argv), collect=collect_fn, out=out)
        return code, out.getvalue()

    def test_success_output(self):
        code, out = self.run_main(lambda t: collect(GOOD))
        self.assertEqual(code, 0)
        d = osn.parse_snapshot_output(out)
        self.assertAlmostEqual(d['yaw'], 0.3, places=12)

    def test_failures_have_no_snapshot_line(self):
        cases = {'NO_MESSAGE': 7, 'INVALID': 8, 'STALE': 9, 'FUTURE_STAMP': 9, 'TIME_INCONSISTENT': 9,
                 'NO_CLOCK': 9, 'CLOCK_NOT_ADVANCING': 9}
        for state, want in cases.items():
            with self.subTest(state=state):
                code, out = self.run_main(lambda t, s=state: ('FAIL', s, 'x'))
                self.assertEqual(code, want)
                self.assertNotIn(osn.LINE_TAG, out)
                with self.assertRaises(osn.SnapshotUnavailable):
                    osn.parse_snapshot_output(out)

    def test_exceptions(self):
        def interrupt(t):
            raise KeyboardInterrupt

        def no_ros(t):
            raise ImportError('No module named rclpy')
        self.assertEqual(self.run_main(interrupt)[0], 11)
        self.assertEqual(self.run_main(no_ros)[0], 6)

    def test_bad_timeout_argument(self):
        for v in ('0', '-1', 'nan', '100'):
            with self.subTest(v=v):
                with self.assertRaises(SystemExit) as cm:
                    import contextlib
                    with contextlib.redirect_stderr(io.StringIO()):
                        osn.main(['--timeout', v], collect=lambda t: collect(GOOD), out=io.StringIO())
                self.assertEqual(cm.exception.code, 2)

    def test_ros_is_not_imported(self):
        self.assertNotIn('rclpy', sys.modules)
        with open(osn.__file__, encoding='utf-8') as f:
            src = f.read()
        for word in ('cmd_vel\'', 'create_publisher', 'Twist'):
            self.assertNotIn(word, src)


def good_output(true_yaw=0.3, **over):
    """成功時の出力。true_yaw で四元数と yaw をそろえ、over で項目を上書きする（不正な出力を作るため）。"""
    snap = collect(GOOD)[1]
    d = json.loads(osn.format_line(snap)[len(osn.LINE_TAG):])
    qx, qy, qz, qw = quat(true_yaw)
    d.update(qx=qx, qy=qy, qz=qz, qw=qw, yaw=true_yaw)
    d.update(over)
    return f'位置 ...\n{osn.LINE_TAG}{json.dumps(d)}\n結果: OK\n'


class TestProvider(unittest.TestCase):
    def provider(self, result):
        calls = []

        def run_child(cmd):
            calls.append(cmd)
            return result
        return osn.make_heading_provider(run_child, 'python3', '/x/odom_snapshot.py'), calls

    def test_ok(self):
        h, calls = self.provider(mr.ChildResult(0, good_output(-2.5926)))
        self.assertAlmostEqual(h(1), -2.5926, places=9)
        self.assertEqual(calls, [['python3', '-B', '/x/odom_snapshot.py']])

    def test_rejections(self):
        line = good_output()
        cases = {
            'exit 9': mr.ChildResult(9, '結果: STALE\n'),
            'exit 0 but not OK': mr.ChildResult(0, line.replace('結果: OK', '結果: STALE')),
            'not started': mr.ChildResult(None, '', started=False),
            'unknown exit': mr.ChildResult(None, line),
            'no line': mr.ChildResult(0, '結果: OK\n'),
            'two lines': mr.ChildResult(0, line.replace('結果: OK', line.splitlines()[1] + '\n結果: OK')),
            'bad json': mr.ChildResult(0, f'{osn.LINE_TAG}{{broken\n結果: OK\n'),
            'yaw mismatch': mr.ChildResult(0, good_output(0.3, yaw=0.4)),
            'nan': mr.ChildResult(0, good_output(0.3, x=float('nan'))),
            'stale age': mr.ChildResult(0, good_output(0.3, age=0.6)),
            'negative age': mr.ChildResult(0, good_output(0.3, age=-0.1)),
            'one sample': mr.ChildResult(0, good_output(0.3, n_samples=1)),
            'missing key': mr.ChildResult(0, good_output(0.3).replace('"clock"', '"clk"')),
            'bad quaternion': mr.ChildResult(0, good_output(0.3, qw=2.0)),
        }
        for name, res in cases.items():
            with self.subTest(case=name):
                h, _ = self.provider(res)
                with self.assertRaises(osn.SnapshotUnavailable):
                    h(1)


class TestWithMissionRunner(RunnerCase):
    """mission_runner.py を変更せず、heading に make_heading_provider() を渡して遷移を確かめる。"""

    def test_runner_uses_provider_value(self):
        h = osn.make_heading_provider(lambda cmd: mr.ChildResult(0, good_output(0.0)))
        ex = FakeExecutor(dryrun=[DRY_OK], search=[(0, [located(3)]), (0, [located(3)])])
        r = self.runner(ex, heading=h)
        self.assertEqual(r.run(), 'END_REPORTED')
        self.assertEqual([x.psi_rad for x in r.rounds], [0.0, 0.0])

    def test_runner_undetermined_on_failure(self):
        h = osn.make_heading_provider(lambda cmd: mr.ChildResult(9, '結果: STALE\n'))
        ex = FakeExecutor(dryrun=[DRY_OK], search=[(10, [no_person(3)])])
        r = self.runner(ex, heading=h)
        self.assertEqual(r.run(), 'END_UNDETERMINED')
        self.assertEqual(r.rounds, [])
        self.assertEqual(ex.count('spin'), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
