#!/usr/bin/env python3
"""test_rescue_mission.py : rescue_mission.py（計画モード）の単体テスト（STEP 19-2）。

ROS 2・Gazebo・YOLO は使わない。/cmd_vel は扱わない。
合成データは loader を差し替えて与える（ファイルを作らない）。--out の成功例だけ、一時ディレクトリに書いてすぐ消す。
実データの確認は results/step15-17/ の CSV を読むだけで、無ければスキップする。
実行: python3 -B test_rescue_mission.py
"""
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mission_logic as ml  # noqa: E402
import rescue_mission as rm  # noqa: E402
from test_mission_logic import located, no_person, row  # noqa: E402

DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'step15-17'))


def run(argv, files=None):
    """(終了コード, 標準出力) を返す。files: {パス: 行のリスト または 例外}。"""
    def loader(path):
        v = files[path]
        if isinstance(v, Exception):
            raise v
        return v
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            code = rm.main(argv, loader if files is not None else ml.load_search_csv, out)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue() + err.getvalue()


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def rounds_args(spec):
    """spec: [(id, psi, spin), ...] → --round 引数（CSV のパスは id.csv）"""
    a = ['--plan']
    for rid, psi, spin in spec:
        a += ['--round', rid, f'{rid}.csv', str(psi), spin]
    return a


class TestArguments(unittest.TestCase):
    def test_missing_arguments(self):
        for argv in ([], ['--plan'], ['--round', 'a', 'a.csv', '0', 'stay'], ['--plan', '--round', 'a', 'a.csv', '0']):
            with self.subTest(argv=argv):
                self.assertEqual(run(argv, {})[0], 2)

    def test_bad_pose_or_spin(self):
        for psi, spin in (('abc', 'stay'), ('4.0', 'stay'), ('nan', 'stay'), ('0', 'yes'), ('0', 'True')):
            with self.subTest(psi=psi, spin=spin):
                self.assertEqual(run(['--plan', '--round', 'a', 'a.csv', psi, spin], {})[0], 2)

    def test_no_run_mode(self):
        self.assertEqual(run(['--plan', '--run'] + rounds_args([('a', 0, 'stay')])[1:], {})[0], 2)


class TestDecisions(unittest.TestCase):
    def test_reported(self):
        code, out = run(rounds_args([('a', 0.0, 'stay'), ('b', 0.0, 'stay')]),
                        {'a.csv': located(3), 'b.csv': located(3)})
        self.assertEqual(code, 0)
        self.assertIn('判定: DONE_REPORTED', out)
        self.assertIn('REPORTED', out)
        self.assertIn('別の観測ラウンドでも再観測', out)
        self.assertIn('実際の要救助者の発見や位置の正しさを保証しない', out)

    def test_research(self):
        code, out = run(rounds_args([('a', 0.0, 'stay')]), {'a.csv': located(3)})
        self.assertEqual(code, 0)
        self.assertIn('判定: RESEARCH', out)
        self.assertIn('CANDIDATE', out)

    def test_spin_next_and_no_person_meaning(self):
        code, out = run(rounds_args([('a', 0.0, 'stay')]), {'a.csv': no_person(3)})
        self.assertEqual(code, 0)
        self.assertIn('判定: SPIN_NEXT', out)
        self.assertIn('その向きに人物がいないことは意味しない', out)

    def test_done_not_found(self):
        hs = [ml.wrap(0.92 * i) for i in range(8)]
        spec = [(f'r{i}', h, 'stay' if i == 0 else 'spin') for i, h in enumerate(hs)]
        files = {f'r{i}.csv': no_person(3, psi=h) for i, h in enumerate(hs)}
        code, out = run(rounds_args(spec), files)
        self.assertEqual(code, 0)
        self.assertIn('判定: DONE_NOT_FOUND', out)
        self.assertIn('一周の判定: 成立', out)
        self.assertIn('不在の証明ではない', out)

    def test_abnormal_state_is_not_normal_no_person(self):
        rows = no_person(2) + [row('ODOM_TIME_MISMATCH', 3)]
        code, out = run(rounds_args([('a', 0.0, 'stay')]), {'a.csv': rows})
        self.assertEqual(code, 0)
        self.assertIn('判定: RESEARCH', out)
        self.assertIn('observed=no', out)
        self.assertIn('正常な未検出ではない', out)

    def test_empty_data(self):
        code, out = run(rounds_args([('a', 0.0, 'stay')]), {'a.csv': []})
        self.assertEqual(code, 0)
        self.assertIn('判定: RESEARCH', out)
        self.assertIn('行なし', out)

    def test_unreadable_csv_aborts(self):
        code, out = run(rounds_args([('a', 0.0, 'stay')]), {'a.csv': ml.InputError('列が足りない')})
        self.assertEqual(code, 4)
        self.assertIn('判定: ABORT', out)
        code, _ = run(rounds_args([('a', 0.0, 'stay')]), {'a.csv': FileNotFoundError('a.csv')})
        self.assertEqual(code, 4)

    def test_pose_mismatch_aborts(self):
        code, out = run(rounds_args([('a', 0.3, 'stay')]), {'a.csv': no_person(3, psi=0.0)})
        self.assertEqual(code, 4)
        self.assertIn('判定: ABORT', out)

    def test_order_contradiction_aborts(self):
        code, _ = run(rounds_args([('a', 0.92, 'stay'), ('b', 0.0, 'spin')]),
                      {'a.csv': no_person(3, psi=0.92), 'b.csv': no_person(3, psi=0.0)})
        self.assertEqual(code, 4)

    def test_duplicate_csv_aborts(self):
        argv = ['--plan', '--round', 'a', 'x.csv', '0', 'stay', '--round', 'b', 'x.csv', '0', 'stay']
        self.assertEqual(run(argv, {'x.csv': located(3)})[0], 4)

    def test_unknown_pose_or_order_is_undetermined(self):
        files = {'a.csv': no_person(3), 'b.csv': no_person(3, psi=0.92)}
        for spec in ([('a', 0.0, 'stay'), ('b', '?', 'spin')], [('a', 0.0, 'stay'), ('b', 0.92, '?')]):
            with self.subTest(spec=spec):
                code, out = run(rounds_args(spec), files)
                self.assertEqual(code, 3)
                self.assertIn('一周の判定: 判定できない', out)
        code, _ = run(rounds_args([('a', 0.0, 'stay'), ('b', 0.92, 'spin')]) + ['--unordered'], files)
        self.assertEqual(code, 3)

    def test_expect_cumulative(self):
        files = {'a.csv': no_person(3), 'b.csv': no_person(3, psi=0.92)}
        argv = rounds_args([('a', 0.0, 'stay'), ('b', 0.92, 'spin')])
        self.assertEqual(run(argv + ['--expect-cumulative', '0.92'], files)[0], 0)
        code, out = run(argv + ['--expect-cumulative', '1.5'], files)
        self.assertEqual(code, 4)
        self.assertIn('累積角の計算値', out)

    def test_unverified_distance_note_is_shown(self):
        code, out = run(rounds_args([('a', 0.0, 'stay'), ('b', 0.0, 'stay')]),
                        {'a.csv': located(3, z_a=4.5), 'b.csv': located(3, z_a=4.5)})
        self.assertEqual(code, 0)
        self.assertIn('未検証距離', out)


class TestOutputFile(unittest.TestCase):
    def test_existing_file_is_not_overwritten(self):
        target = os.path.abspath(__file__)
        before = sha256(target)
        code, out = run(rounds_args([('a', 0.0, 'stay')]) + ['--out', target], {'a.csv': no_person(3)})
        self.assertEqual(code, 12)
        self.assertEqual(sha256(target), before)

    def test_input_csv_path_is_refused(self):
        code, _ = run(rounds_args([('a', 0.0, 'stay')]) + ['--out', 'a.csv'], {'a.csv': no_person(3)})
        self.assertEqual(code, 12)

    def test_new_file_is_written(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'plan.json')
            code, _ = run(rounds_args([('a', 0.0, 'stay')]) + ['--out', p], {'a.csv': no_person(3)})
            self.assertEqual(code, 0)
            with open(p, encoding='utf-8') as f:
                data = json.load(f)
            self.assertEqual(data['decision'], 'SPIN_NEXT')
            self.assertEqual(data['n_rounds'], 1)
            self.assertTrue(data['full_turn_judgeable'])


@unittest.skipUnless(os.path.isdir(DATA), 'results/step15-17 がない')
class TestRealData(unittest.TestCase):
    def csv(self, tag):
        return os.path.join(DATA, f'search_20261024_{tag}.csv')

    def test_t2_s1_reported(self):
        code, out = run(['--plan', '--round', 'T2', self.csv('164652'), '0.0', 'stay',
                         '--round', 'S1', self.csv('170016'), '0.0', 'stay'])
        self.assertEqual(code, 0)
        self.assertIn('判定: DONE_REPORTED', out)

    def test_spin_sequence_to_k6(self):
        seq = [('S4', '171320', 0.9224), ('k1', '173859', 1.8453), ('k2', '174245', 2.7681),
               ('k3', '174910', -2.5926), ('k4', '175222', -1.6796), ('k5', '175508', -0.7668),
               ('k6', '175735', 0.1555)]
        argv = ['--plan']
        for i, (rid, tag, psi) in enumerate(seq):
            argv += ['--round', rid, self.csv(tag), str(psi), 'stay' if i == 0 else 'spin']
        code, out = run(argv + ['--expect-cumulative', '5.5167'])
        self.assertEqual(code, 0)
        self.assertIn('判定: RESEARCH', out)
        code, out = run(argv[:1 + 5 * 6])            # S4〜k5（1 ラウンド = 5 個の引数）
        self.assertEqual(code, 0)
        self.assertIn('判定: SPIN_NEXT', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
