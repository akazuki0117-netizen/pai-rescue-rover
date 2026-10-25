#!/usr/bin/env python3
"""person_geometry.py : 人物の検出枠から方向・距離・ワールド座標を求める純粋な関数。

locate_once.py（SHA-256 83e84dbc…53e3b0）の main() にある計算を、式・条件・定数・計算順序を変えずに移したもの。
ROS 2 と YOLO には依存しない。途中の値は丸めない（丸めるのは表示側だけ）。

座標系: 画像 u 右正・v 下正 / ロボット（base_footprint）x 前・y 左 / ワールド = /odom
前提: カメラは base_footprint から (CAM_X, 0, CAM_Z)、回転なし（水平）。床は平坦。人物は床に立っている。
"""
import math

CAM_X = 0.08             # m  STEP 13 の gz model -i で 0.080 を確認
CAM_Z = 1.04             # m  STEP 13 の gz model -i で 1.040 を確認
EDGE_PX = 2.0            # 画像の端から何ピクセル以内を「端に接している」とみなすか（暫定）
MIN_BELOW_HORIZON = 5.0  # y2 - cy がこれ未満なら距離 A を出さない（暫定）


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def edge_flags(x1, y1, x2, y2, w, h, cy):
    flags = []
    if y2 >= h - 1 - EDGE_PX:
        flags.append('BOTTOM_CUT')
    if y1 <= EDGE_PX:
        flags.append('TOP_CUT')
    if x1 <= EDGE_PX or x2 >= w - 1 - EDGE_PX:
        flags.append('SIDE_CUT')
    if 'BOTTOM_CUT' not in flags and (y2 - cy) < MIN_BELOW_HORIZON:
        flags.append('ABOVE_HORIZON')
    return flags


def bearing(x1, x2, cx, fx):
    """(u_c, phi) を返す。phi は左が正、右が負 [rad]。"""
    u_c = 0.5 * (x1 + x2)
    phi = math.atan2(cx - u_c, fx)
    return u_c, phi


def distance_foot(y2, cy, fy, flags):
    """距離 A（足元）。確定しない条件なら None。"""
    if 'BOTTOM_CUT' in flags or 'ABOVE_HORIZON' in flags:
        return None
    return fy * CAM_Z / (y2 - cy)


def distance_height(y1, y2, fy, height, flags):
    """距離 B（枠の高さ、身長の仮定による参考値）。無効なら None。"""
    if 'BOTTOM_CUT' in flags or 'TOP_CUT' in flags:
        return None
    return fy * height / (y2 - y1)


def to_robot(z_a, u_c, cx, fx):
    px = CAM_X + z_a
    py = z_a * (cx - u_c) / fx
    return px, py


def to_world(px, py, xr, yr, psi):
    X = xr + px * math.cos(psi) - py * math.sin(psi)
    Y = yr + px * math.sin(psi) + py * math.cos(psi)
    return X, Y


def locate(box, w, h, fx, fy, cx, cy, xr, yr, psi, height=1.7):
    """locate_once.py と同じ順序で計算し、結果を dict で返す。z_a が None なら座標は None。"""
    x1, y1, x2, y2 = box
    flags = edge_flags(x1, y1, x2, y2, w, h, cy)
    u_c, phi = bearing(x1, x2, cx, fx)
    z_a = distance_foot(y2, cy, fy, flags)
    z_b = distance_height(y1, y2, fy, height, flags)
    px = py = X = Y = None
    if z_a is not None:
        px, py = to_robot(z_a, u_c, cx, fx)
        X, Y = to_world(px, py, xr, yr, psi)
    return {'flags': flags, 'u_c': u_c, 'phi': phi, 'z_a': z_a, 'z_b': z_b,
            'px': px, 'py': py, 'X': X, 'Y': Y}
