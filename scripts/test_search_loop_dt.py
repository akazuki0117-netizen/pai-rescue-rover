#!/usr/bin/env python3
"""test_search_loop_dt.py : search_loop.one_cycle() の時刻差の判定を、モックで確かめる単体テスト。

本番コードは変更しない。ROS 2・Gazebo・YOLO は起動・読み込みしない。/cmd_vel は扱わない。
境界の値は 2 進数で誤差なく表せる値（101.0, 100.75, 0.25 など）を使い、実際に比較される dt を表示する。
期待どおりでない場合や想定外の例外は FAIL とし、1件でも FAIL があれば終了コード 1。
"""
import os
import sys
import time
import traceback
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search_loop as sl  # noqa: E402

INFO_K = (530.467, 530.467, 320.5, 240.5)       # (fx, fy, cx, cy) T1 と同じ
P4_BOX = [343.1, 122.4, 456.3, 436.8]           # T1 と同じ（正常系の対照用）


class FakeRx:
    def __init__(self, t_img_sec, odom):
        self.msg = NS(header=NS(stamp=NS(sec=t_img_sec, nanosec=0)), width=640, height=480)
        self.odom = odom                        # (t, x, y, psi) または None

    def latest_image(self):
        return self.msg, 1

    def nearest_odom(self, t):
        return self.odom


class FakeBridge:
    def __init__(self):
        self.calls = 0

    def imgmsg_to_cv2(self, msg, enc):
        self.calls += 1
        return object()


class FakeTensor:
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return self.v


class FakeModel:
    def __init__(self, boxes):              # boxes: [(conf, [x1, y1, x2, y2]), ...]
        self.calls = 0
        self.boxes = boxes

    def predict(self, img, **kw):
        self.calls += 1
        b = NS(conf=FakeTensor([c for c, _ in self.boxes]), xyxy=FakeTensor([bb for _, bb in self.boxes]))
        return [NS(boxes=b)]


def run_case(name, t_img_sec, odom, max_dt, boxes, expect):
    rx, bridge, model = FakeRx(t_img_sec, odom), FakeBridge(), FakeModel(boxes)
    clusters = sl.PositionClusters(0.5)
    args = NS(timeout=0.2, max_dt=max_dt, conf=0.25, height=1.7)
    dt = None if odom is None else sl.stamp_sec(rx.msg.header) - odom[0]   # one_cycle と同じ式
    if dt is None:
        branch = 'NO_ODOM'
    else:
        branch = 'ODOM_TIME_MISMATCH' if abs(dt) > max_dt else '通常処理へ'
    try:
        row = sl.one_cycle(1, time.monotonic(), rx, {'last_seq': 0}, args, model, bridge, clusters, INFO_K)
    except Exception:  # noqa: BLE001
        print(f'FAIL {name}: 想定外の例外（dt={dt!r} max_dt={max_dt} 期待する分岐={branch}）')
        traceback.print_exc()
        return False
    got = {'status': row['status'], 'predict': model.calls, 'bridge': bridge.calls,
           'X': row['X'], 'Y': row['Y'], 'cluster_id': row['cluster_id'], 'n_clusters': len(clusters.clusters)}
    ok = all(got[k] == v for k, v in expect.items())
    if dt is not None:                       # 実際の dt から期待される分岐と、実際の状態の整合
        ok = ok and ((row['status'] == 'ODOM_TIME_MISMATCH') == (abs(dt) > max_dt))
    print(f'{"PASS" if ok else "FAIL"} {name}')
    print(f'      dt={dt!r} max_dt={max_dt} 期待する分岐={branch} 実際の状態={row["status"]}')
    print(f'      got={got}')
    if not ok:
        print(f'      expect={expect}')
    return ok


EMPTY = {'X': '', 'Y': '', 'cluster_id': '', 'n_clusters': 0}
CASES = [
    ('dt=+0.5 → MISMATCH', 101, (100.5, -2.0, -0.5, 0.0), 0.1, [],
     dict(status='ODOM_TIME_MISMATCH', predict=0, bridge=0, **EMPTY)),
    ('dt=-0.5 → MISMATCH', 101, (101.5, -2.0, -0.5, 0.0), 0.1, [],
     dict(status='ODOM_TIME_MISMATCH', predict=0, bridge=0, **EMPTY)),
    ('dt=0.05 → 通常処理（人物なし）', 101, (100.95, -2.0, -0.5, 0.0), 0.1, [],
     dict(status='NO_PERSON', predict=1, bridge=1, **EMPTY)),
    ('odom なし → NO_ODOM', 101, None, 0.1, [],
     dict(status='NO_ODOM', predict=0, bridge=0, **EMPTY)),
    ('境界 dt=0.25 == max_dt=0.25 → 通常処理', 101, (100.75, -2.0, -0.5, 0.0), 0.25, [],
     dict(status='NO_PERSON', predict=1, bridge=1, **EMPTY)),
    ('境界 dt=0.25 > max_dt=0.2499 → MISMATCH', 101, (100.75, -2.0, -0.5, 0.0), 0.2499, [],
     dict(status='ODOM_TIME_MISMATCH', predict=0, bridge=0, **EMPTY)),
    ('対照 dt=0 → LOCATED（クラスタ1）', 101, (101.0, -2.0, -0.5, 0.0), 0.1, [(0.904, P4_BOX)],
     dict(status='LOCATED', predict=1, bridge=1, X='0.8904', Y='-0.9196', cluster_id=1, n_clusters=1)),
]

if __name__ == '__main__':
    results = [run_case(*c) for c in CASES]
    n_fail = results.count(False)
    print(f'合計 {len(results)} 件: PASS {results.count(True)} / FAIL {n_fail}')
    sys.exit(1 if n_fail else 0)
