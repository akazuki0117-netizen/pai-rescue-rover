#!/usr/bin/env python3
"""test_mission_logic.py : mission_logic.py の判定を確かめる単体テスト（STEP 19-1）。

ROS 2・Gazebo・YOLO は起動・読み込みしない。/cmd_vel は扱わない。ファイルは書かない（CSV は io.StringIO で与える）。
実データの再生テストは results/step15-17/ の CSV を読むだけで、無ければスキップする。
実行: python3 -B test_mission_logic.py
"""
import io
import itertools
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mission_logic as ml  # noqa: E402
from search_loop import RAW_FIELDS  # noqa: E402

DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'step15-17'))
_stamp = itertools.count(1000)
XR, YR = -2.0, -0.5


def row(status, cycle, psi=0.0, X=0.89, Y=-0.86, conf=0.92, z_a=2.80, flags='', **kw):
    """search_loop.py と同じ 25 列の行を作る。psi は rad。"""
    r = {k: '' for k in RAW_FIELDS}
    r.update(cycle=str(cycle), status=status, img_stamp=f'{next(_stamp)}.000')
    if status in ('LOCATED', 'BEARING_ONLY', 'NO_PERSON'):
        r.update(xr=f'{XR:.4f}', yr=f'{YR:.4f}', psi_deg=f'{math.degrees(psi):.3f}', n_det='0')
    if status in ('LOCATED', 'BEARING_ONLY'):
        r.update(n_det='1', conf=f'{conf}', u_c='388.4', phi_deg='-7.5', flags=flags)
    if status == 'LOCATED':
        r.update(z_a=f'{z_a}', X=repr(X), Y=repr(Y))
    r.update(kw)
    return r


def rnd(rid, rows, order=None, psi=None, spun=None, source=None):
    return ml.RoundInput(round_id=rid, rows=rows, order=order, psi_rad=psi, after_spin=spun, source=source)


def located(n, psi=0.0, **kw):
    return [row('LOCATED', i + 1, psi=psi, **kw) for i in range(n)]


def no_person(n, psi=0.0):
    return [row('NO_PERSON', i + 1, psi=psi) for i in range(n)]


def states(ev):
    return [c.state for c in ev.clusters]


class TestDetectionLevels(unittest.TestCase):
    def test_single_detection_is_not_reported(self):
        ev = ml.evaluate([rnd('a', located(1) + no_person(2)[1:], 1, 0.0, False)])
        self.assertEqual(states(ev), ['DETECTED'])
        self.assertNotEqual(ev.decision, 'DONE_REPORTED')
        self.assertEqual(ev.decision, 'RESEARCH')

    def test_many_frames_in_one_round_are_not_reported(self):
        ev = ml.evaluate([rnd('a', located(10), 1, 0.0, False)])
        self.assertEqual(states(ev), ['CANDIDATE'])
        self.assertEqual(ev.decision, 'RESEARCH')
        self.assertTrue(any('再観測がない' in n for n in ev.clusters[0].notes))

    def test_reobserved_in_another_round_is_reported(self):
        ev = ml.evaluate([rnd('a', located(3), 1, 0.0, False), rnd('b', located(3), 2, 0.0, False)])
        self.assertEqual(states(ev), ['REPORTED'])
        self.assertEqual(ev.decision, 'DONE_REPORTED')

    def test_second_round_with_one_detection_but_total_enough(self):
        ev = ml.evaluate([rnd('a', located(2), 1, 0.0, False), rnd('b', located(1), 2, 0.0, False)])
        self.assertEqual(states(ev), ['REPORTED'])

    def test_reobservation_far_away_is_another_cluster(self):
        ev = ml.evaluate([rnd('a', located(3), 1, 0.0, False),
                          rnd('b', located(3, X=0.89 + 0.6), 2, 0.0, False)])
        self.assertEqual(states(ev), ['CANDIDATE', 'CANDIDATE'])
        self.assertEqual(ev.decision, 'RESEARCH')

    def test_research_limit_then_spin(self):
        rs = [rnd(f'r{i}', located(3, X=0.89 + 0.6 * i), i + 1, 0.0, False) for i in range(3)]
        ev = ml.evaluate(rs)
        self.assertEqual(ev.decision, 'SPIN_NEXT')


class TestExclusion(unittest.TestCase):
    def test_low_confidence_is_recorded_but_not_evidence(self):
        ev = ml.evaluate([rnd('a', located(3, conf=0.49), 1, 0.0, False),
                          rnd('b', located(3, conf=0.49), 2, 0.0, False)])
        self.assertEqual(states(ev), ['DETECTED'])
        self.assertEqual(len(ev.clusters[0].records), 6)
        self.assertEqual(ev.decision, 'SPIN_NEXT')

    def test_confidence_exactly_threshold_counts(self):
        ev = ml.evaluate([rnd('a', located(3, conf=0.5), 1, 0.0, False),
                          rnd('b', located(3, conf=0.5), 2, 0.0, False)])
        self.assertEqual(states(ev), ['REPORTED'])

    def test_each_edge_flag_is_excluded(self):
        for f in ('SIDE_CUT', 'TOP_CUT', 'BOTTOM_CUT', 'ABOVE_HORIZON', 'SIDE_CUT|TOP_CUT'):
            with self.subTest(flag=f):
                ev = ml.evaluate([rnd('a', located(3, flags=f), 1, 0.0, False),
                                  rnd('b', located(3, flags=f), 2, 0.0, False)])
                self.assertEqual(states(ev), ['DETECTED'])
                self.assertEqual(len(ev.clusters[0].evidence), 0)
                self.assertEqual(len(ev.clusters[0].records), 6)

    def test_flagged_rows_do_not_complete_a_report(self):
        ev = ml.evaluate([rnd('a', located(3), 1, 0.0, False),
                          rnd('b', located(3, flags='SIDE_CUT'), 2, 0.0, False)])
        self.assertEqual(states(ev), ['CANDIDATE'])
        self.assertEqual(len(ev.clusters[0].records), 3)

    def test_unverified_distance_is_kept_with_note(self):
        ev = ml.evaluate([rnd('a', located(3, z_a=4.5), 1, 0.0, False),
                          rnd('b', located(3, z_a=4.5), 2, 0.0, False)])
        self.assertEqual(states(ev), ['REPORTED'])
        self.assertTrue(any('未検証距離' in n for n in ev.clusters[0].notes))
        self.assertTrue(all(o.unverified_distance for o in ev.clusters[0].evidence))

    def test_verified_distance_has_no_note(self):
        ev = ml.evaluate([rnd('a', located(3, z_a=2.8), 1, 0.0, False)])
        self.assertFalse(any('未検証距離' in n for n in ev.clusters[0].notes))


class TestNoPersonVsAbnormal(unittest.TestCase):
    def test_no_person_is_normal_observation(self):
        ev = ml.evaluate([rnd('a', no_person(3), 1, 0.0, False)])
        self.assertTrue(ev.rounds[0].observed)
        self.assertEqual(ev.decision, 'SPIN_NEXT')
        self.assertTrue(any('いないことを意味しない' in n for n in ev.notes))

    def test_abnormal_statuses_are_not_observations(self):
        for st in ('ODOM_TIME_MISMATCH', 'CYCLE_ERROR', 'NO_IMAGE', 'NO_ODOM'):
            with self.subTest(status=st):
                bad = row(st, 3, flags='RuntimeError: x' if st == 'CYCLE_ERROR' else '')
                ev = ml.evaluate([rnd('a', no_person(2) + [bad], 1, 0.0, False)])
                self.assertFalse(ev.rounds[0].observed)
                self.assertEqual(ev.decision, 'RESEARCH')

    def test_repeated_abnormal_rounds_abort(self):
        rs = [rnd(f'r{i}', [row('ODOM_TIME_MISMATCH', c) for c in (1, 2, 3)], i + 1, 0.0, False)
              for i in range(3)]
        ev = ml.evaluate(rs)
        self.assertEqual(ev.decision, 'ABORT')

    def test_too_few_inference_cycles(self):
        ev = ml.evaluate([rnd('a', no_person(1), 1, 0.0, False)])
        self.assertFalse(ev.rounds[0].observed)
        self.assertEqual(ev.decision, 'RESEARCH')


def spin_sequence(headings, empty_at=()):
    rs = []
    for i, h in enumerate(headings):
        rows = [row('NO_PERSON', 1, psi=h)] if i in empty_at else no_person(3, psi=h)
        rs.append(rnd(f'r{i}', rows, i + 1, h, i > 0))
    return rs


class TestRotation(unittest.TestCase):
    def test_accumulation_across_pi(self):
        hs = [2.7681, -2.5926, -1.6796]           # STEP 17 の k=2 → k=3 → k=4（±180° をまたぐ）
        ev = ml.evaluate(spin_sequence(hs))
        self.assertAlmostEqual(ev.cumulative_rad, 2 * math.pi + (-1.6796 - 2.7681), places=9)
        self.assertGreater(ev.cumulative_rad, 0)
        self.assertEqual(ev.decision, 'SPIN_NEXT')

    def test_full_turn_gives_not_found(self):
        hs = [ml.wrap(0.92 * i) for i in range(8)]   # 7 回の旋回、累積 6.44 rad
        ev = ml.evaluate(spin_sequence(hs))
        self.assertTrue(ev.full_turn)
        self.assertEqual(ev.decision, 'DONE_NOT_FOUND')
        self.assertTrue(any('不在の証明ではない' in r for r in ev.reasons))

    def test_short_of_full_turn_spins_next(self):
        ev = ml.evaluate(spin_sequence([ml.wrap(0.92 * i) for i in range(7)]))   # 累積 5.52 rad
        self.assertFalse(ev.full_turn)
        self.assertEqual(ev.decision, 'SPIN_NEXT')

    def test_spin_limit(self):
        ev = ml.evaluate(spin_sequence([0.5 * i for i in range(8)]))             # 7 回で 3.5 rad
        self.assertEqual(ev.spins, 7)
        self.assertEqual(ev.decision, 'DONE_SPIN_LIMIT')

    def test_gap_in_observed_headings_is_not_full_turn(self):
        hs = [ml.wrap(0.92 * i) for i in range(9)]
        ev = ml.evaluate(spin_sequence(hs, empty_at=(3,)))   # 4 番目の向きは観測できていない
        self.assertFalse(ev.full_turn)
        self.assertEqual(ev.decision, 'DONE_SPIN_LIMIT')

    def test_missing_order_is_undetermined(self):
        rs = spin_sequence([ml.wrap(0.92 * i) for i in range(8)])
        rs[2].order = None
        ev = ml.evaluate(rs)
        self.assertEqual(ev.decision, 'UNDETERMINED')
        self.assertFalse(ev.full_turn)

    def test_missing_heading_or_spin_flag_is_undetermined(self):
        for attr in ('psi_rad', 'after_spin'):
            with self.subTest(attr=attr):
                rs = spin_sequence([ml.wrap(0.92 * i) for i in range(8)])
                setattr(rs[5], attr, None)
                self.assertEqual(ml.evaluate(rs).decision, 'UNDETERMINED')

    def test_report_without_order_is_still_reported(self):
        ev = ml.evaluate([rnd('a', located(3)), rnd('b', located(3))])
        self.assertEqual(ev.decision, 'DONE_REPORTED')

    def test_non_consecutive_order_aborts(self):
        rs = spin_sequence([0.0, 0.92])
        rs[1].order = 3
        self.assertEqual(ml.evaluate(rs).decision, 'ABORT')

    def test_backward_or_too_large_spin_aborts(self):
        for hs in ([0.92, 0.0], [0.0, 1.2], [0.0, 0.0]):
            with self.subTest(headings=hs):
                self.assertEqual(ml.evaluate(spin_sequence(hs)).decision, 'ABORT')

    def test_heading_change_without_spin_aborts(self):
        rs = [rnd('a', no_person(3, 0.0), 1, 0.0, False), rnd('b', no_person(3, 0.3), 2, 0.3, False)]
        self.assertEqual(ml.evaluate(rs).decision, 'ABORT')

    def test_given_heading_must_match_csv(self):
        ev = ml.evaluate([rnd('a', no_person(3, psi=0.0), 1, 0.1, False)])
        self.assertEqual(ev.decision, 'ABORT')


class TestInvalidInput(unittest.TestCase):
    HEADER = ','.join(RAW_FIELDS)

    def test_empty_round_list(self):
        self.assertEqual(ml.evaluate([]).decision, 'ABORT')

    def test_empty_file_and_missing_columns(self):
        with self.assertRaises(ml.InputError):
            ml.read_search_csv(io.StringIO(''), 'empty')
        with self.assertRaises(ml.InputError):
            ml.read_search_csv(io.StringIO('cycle,status\n1,NO_PERSON\n'), 'short')

    def test_row_with_wrong_column_count(self):
        with self.assertRaises(ml.InputError):
            ml.read_search_csv(io.StringIO(self.HEADER + '\n1,2,3\n'), 'ragged')
        with self.assertRaises(ml.InputError):
            ml.read_search_csv(io.StringIO(self.HEADER + '\n' + ','.join(['1'] * (len(RAW_FIELDS) + 1)) + '\n'), 'long')

    def test_header_only_file_is_not_an_observation(self):
        rows = ml.read_search_csv(io.StringIO(self.HEADER + '\n'), 'header-only')
        self.assertEqual(rows, [])
        ev = ml.evaluate([rnd('a', rows, 1, 0.0, False)])
        self.assertFalse(ev.rounds[0].observed)
        self.assertNotIn(ev.decision, ('DONE_NOT_FOUND', 'DONE_REPORTED', 'SPIN_NEXT'))

    def test_bad_values_abort(self):
        cases = [dict(X='abc'), dict(Y=''), dict(z_a='nan'), dict(conf='1.5'), dict(z_a='-1'),
                 dict(status='FOUND'), dict(cycle='x'), dict(psi_deg='inf')]
        for kw in cases:
            with self.subTest(**kw):
                rows = located(3)
                rows[1].update(kw)
                ev = ml.evaluate([rnd('a', rows, 1, 0.0, False), rnd('b', located(3), 2, 0.0, False)])
                self.assertEqual(ev.decision, 'ABORT')
                self.assertTrue(ev.errors)

    def test_missing_column_in_row_dict_aborts(self):
        rows = located(3)
        del rows[0]['X']
        self.assertEqual(ml.evaluate([rnd('a', rows, 1, 0.0, False)]).decision, 'ABORT')

    def test_duplicates_abort(self):
        a = located(3)
        dup_cycle = located(3)
        dup_cycle[2]['cycle'] = '1'
        cases = {
            'round_id': [rnd('a', located(3), 1, 0.0, False), rnd('a', located(3), 2, 0.0, False)],
            'same rows twice': [rnd('a', a, 1, 0.0, False), rnd('b', [dict(r) for r in a], 2, 0.0, False)],
            'same file twice': [rnd('a', located(3), 1, 0.0, False, 'x.csv'),
                                rnd('b', located(3), 2, 0.0, False, 'x.csv')],
            'cycle': [rnd('a', dup_cycle, 1, 0.0, False)],
        }
        for name, rs in cases.items():
            with self.subTest(case=name):
                ev = ml.evaluate(rs)
                self.assertEqual(ev.decision, 'ABORT')
                self.assertFalse(any(c.state == 'REPORTED' for c in ev.clusters))

    def test_robot_moved_aborts(self):
        rows = no_person(3)
        rows[2]['xr'] = f'{XR + 0.06:.4f}'
        self.assertEqual(ml.evaluate([rnd('a', rows, 1, 0.0, False)]).decision, 'ABORT')


class TestThresholds(unittest.TestCase):
    def test_cluster_radius_boundary(self):
        ev = ml.evaluate([rnd('a', [row('LOCATED', 1, X=1.0, Y=-1.0), row('LOCATED', 2, X=1.5, Y=-1.0)], 1, 0.0, False)])
        self.assertEqual(len(ev.clusters), 1)            # ちょうど 0.5 m はまとめる
        ev = ml.evaluate([rnd('a', [row('LOCATED', 1, X=1.0, Y=-1.0), row('LOCATED', 2, X=1.500001, Y=-1.0)], 1, 0.0, False)])
        self.assertEqual(len(ev.clusters), 2)

    def _std_case(self, d):
        a = [row('LOCATED', 1, X=1.0 - d, Y=-0.9), row('LOCATED', 2, X=1.0, Y=-0.9)]
        b = [row('LOCATED', 1, X=1.0 + d, Y=-0.9)]
        return ml.evaluate([rnd('a', a, 1, 0.0, False), rnd('b', b, 2, 0.0, False)])

    def test_std_boundary(self):
        ev = self._std_case(0.15)                        # 標本標準偏差がほぼちょうど 0.15
        self.assertAlmostEqual(ev.clusters[0].std()[0], 0.15, places=9)
        self.assertEqual(states(ev), ['REPORTED'])
        ev = self._std_case(0.151)
        self.assertEqual(states(ev), ['CANDIDATE'])

    def test_wrap(self):
        self.assertAlmostEqual(ml.wrap(math.pi + 0.1), -math.pi + 0.1)
        self.assertAlmostEqual(ml.wrap(-math.pi - 0.1), math.pi - 0.1)


@unittest.skipUnless(os.path.isdir(DATA), 'results/step15-17 がない')
class TestReplaySTEP15to17(unittest.TestCase):
    """実際の CSV を、作業ログに記録した順序と /odom の向き（rad）で与える。"""
    RUNS = [('T2', '164652', 0.0, False), ('S1', '170016', 0.0, False),
            ('S4', '171320', 0.9224, True), ('k1', '173859', 1.8453, True), ('k2', '174245', 2.7681, True),
            ('k3', '174910', -2.5926, True), ('k4', '175222', -1.6796, True), ('k5', '175508', -0.7668, True),
            ('k6', '175735', 0.1555, True)]

    def rounds(self, names):
        out = []
        for i, (rid, tag, psi, spun) in enumerate([r for r in self.RUNS if r[0] in names]):
            path = os.path.join(DATA, f'search_20261024_{tag}.csv')
            out.append(rnd(rid, ml.load_search_csv(path), i + 1, psi, spun if i > 0 else False, path))
        return out

    def test_t2_alone_is_candidate(self):
        ev = ml.evaluate(self.rounds(['T2']))
        self.assertEqual(states(ev), ['CANDIDATE'])
        self.assertEqual(ev.decision, 'RESEARCH')

    def test_t2_then_s1_is_reported(self):
        ev = ml.evaluate(self.rounds(['T2', 'S1']))
        self.assertEqual(states(ev), ['REPORTED'])
        self.assertEqual(ev.decision, 'DONE_REPORTED')
        mx, my = ev.clusters[0].mean()
        self.assertAlmostEqual(mx, 0.8931, places=3)     # T2 10 行と S1 3 行の平均（計算値）
        self.assertEqual(len(ev.clusters[0].evidence), 13)

    def test_spin_sequence_until_k5_spins_next(self):
        ev = ml.evaluate(self.rounds(['S4', 'k1', 'k2', 'k3', 'k4', 'k5']))
        self.assertEqual(ev.clusters, [])
        self.assertEqual(ev.decision, 'SPIN_NEXT')
        self.assertTrue(all(r.observed for r in ev.rounds))

    def test_k6_is_candidate_and_researches(self):
        ev = ml.evaluate(self.rounds(['S4', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6']))
        self.assertEqual(states(ev), ['CANDIDATE'])
        self.assertEqual(ev.decision, 'RESEARCH')
        self.assertAlmostEqual(ev.cumulative_rad, 5.5167, places=3)
        self.assertFalse(ev.full_turn)


if __name__ == '__main__':
    unittest.main(verbosity=2)
