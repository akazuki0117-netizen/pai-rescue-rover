#!/usr/bin/env python3
"""test_mission_runner_allowspin.py : mission_runner.main() の --allow-spin の単体テスト。

ROS 2・Gazebo・YOLO は使わない。MissionRunner は偽物に差し替え、引数の受け渡しだけを確かめる。
実行: python3 -B test_mission_runner_allowspin.py
"""
import contextlib
import inspect
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mission_runner as mr  # noqa: E402


class Spy:
    captured = {}

    def __init__(self, *a, **k):
        Spy.captured = k

    def run(self):
        return 'END_HELD'


def run_main(argv):
    orig = mr.MissionRunner
    mr.MissionRunner = Spy
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = mr.main(argv)
    finally:
        mr.MissionRunner = orig
    return code, dict(Spy.captured), out.getvalue()


class TestAllowSpinFlag(unittest.TestCase):
    def test_default_does_not_enable_spin(self):
        code, kw, out = run_main(['--run'])
        self.assertNotIn('allow_spin', kw)                   # 渡さない → MissionRunner の初期値 False
        self.assertTrue(kw.get('use_odom_snapshot'))
        self.assertEqual(code, mr.EXIT['END_HELD'])
        self.assertNotIn('--allow-spin', out)

    def test_flag_enables_spin(self):
        code, kw, out = run_main(['--run', '--allow-spin'])
        self.assertIs(kw.get('allow_spin'), True)
        self.assertTrue(kw.get('use_odom_snapshot'))
        self.assertIn('旋回は有効', out)

    def test_constructor_default_unchanged(self):
        self.assertFalse(inspect.signature(mr.MissionRunner).parameters['allow_spin'].default)

    def test_run_is_still_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                mr.main(['--allow-spin'])
        self.assertEqual(cm.exception.code, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
