#!/usr/bin/env python3
"""test_mission_runner_approver.py : stdin_approver の入力の扱い（全角の正規化・No の理由の表示）の単体テスト。

ROS 2・Gazebo は使わない。標準入力は読み取り関数の差し替えで与える。
実行: python3 -B test_mission_runner_approver.py
"""
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mission_runner as mr  # noqa: E402


def ask(answer=None, error=None):
    def read_line(timeout):
        if error:
            raise error
        return answer
    with contextlib.redirect_stdout(io.StringIO()) as out:
        ok = mr.stdin_approver('?', 60, read_line=read_line)
    return ok, out.getvalue()


class TestApprover(unittest.TestCase):
    def test_yes_variants(self):
        for a in ('y\n', 'Y\n', 'yes\n', ' YES \n', 'ｙ\n', 'Ｙｅｓ\n'):
            with self.subTest(a=a):
                self.assertIs(ask(a)[0], True)

    def test_no_variants_show_reason(self):
        for a in ('n\n', '\n', 'yy\n', 'はい\n', 'y es\n'):
            with self.subTest(a=a):
                ok, out = ask(a)
                self.assertIs(ok, False)
                self.assertIn('y / yes ではない', out)

    def test_timeout_and_error_show_reason(self):
        ok, out = ask(None)
        self.assertIs(ok, False)
        self.assertIn('時間切れ', out)
        ok, out = ask(error=UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'bad'))
        self.assertIs(ok, False)
        self.assertIn('入力を読めない', out)
        self.assertIn('UnicodeDecodeError', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
