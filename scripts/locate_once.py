#!/usr/bin/env python3
"""locate_once.py : カメラ画像1枚と /odom 1回から、YOLOv8n で検出した人物のワールド座標 (X, Y) を推定する。

しないこと：/cmd_vel の購読・送信、ロボットの移動、画像の保存、画面表示、モデルのダウンロード、インストール。
ただし、ライブラリ（ROS 2・DDS・ultralytics など）によるファイルの書き込みを完全に防げるとは限らない。

実行時に必要な指定（そろっていなければ INIT_ERROR で終了し、ultralytics を import しない）:
  YOLO_CONFIG_DIR=<設定の保存先>  YOLO_OFFLINE=true  YOLO_AUTOINSTALL=false
  PYTHONDONTWRITEBYTECODE=1 または python3 -B

座標系の定義
  画像（ピクセル）: u は右向きが正、v は下向きが正。主点 (cx, cy) は /camera/camera_info の K から読む。
  ロボット座標（base_footprint）: x 前方、y 左、z 上（ROS REP 103）。
  カメラの取り付け: base_footprint から (CAM_X, 0, CAM_Z)、回転なし（正面を水平に見る）。
      STEP 13 の gz model -i で (0.080, 0.000, 1.040) を確認（修正版モデル happy_mini_camfix.sdf）。
  ワールド座標: /odom の位置と向き。STEP 9〜13 で odom の座標系 = ワールドの座標系と見なせることを実測。

計算式（前提：カメラは水平、床は平坦、人物は床の上に立っている）
  方向     phi = atan2(cx - u_c, fx)                     左が正、右が負（u_c = 枠の中心の u）
  距離 A   Z = fy * CAM_Z / (y2 - cy)                     枠の下端 y2 を足元の接地点とみなす（主に使う）
  距離 B   Z = fy * H / (y2 - y1)                         身長 H の仮定による参考値
  ロボット座標  px = CAM_X + Z,  py = Z * (cx - u_c) / fx  （= Z * tan(phi)）
  ワールド座標  X = xr + px*cos(psi) - py*sin(psi)
                Y = yr + px*sin(psi) + py*cos(psi)
  数値例（Z = 2 m, psi = 0, ロボット (0, 0)）:
    u_c = 480（中心より右）→ phi = -16.7°, py = -0.601 → Y < 0（右）
    u_c = 160（中心より左）→ phi = +16.8°, py = +0.604 → Y > 0（左）
  数値例（psi = +90°）: px = 2, py = 0 → (X, Y) = (xr, yr + 2)（ロボットの前方 = ワールドの +y）

距離が確定しない条件
  BOTTOM_CUT : y2 が画像の下端に接している（足元が写っていない）→ 距離 A を出さない
  TOP_CUT    : y1 が画像の上端に接している → 距離 B を出さない
  SIDE_CUT   : x1 または x2 が画像の左右端に接している → 方向も参考扱い（枠の中心がずれる）
  ABOVE_HORIZON : y2 - cy が小さすぎる（足元が地平線付近）→ 距離 A を出さない

結果は最後の「結果: …」の行で判断する。終了コード:
  0=LOCATED（距離 A で座標を確定）  13=BEARING_ONLY（人物は検出したが距離 A を確定できない）
  10=NO_PERSON  2=引数の誤り(argparse)  3=NO_IMAGE  14=NO_ODOM  15=NO_CAMINFO
  4=CONVERT_ERROR  5=MODEL_ERROR  6=INIT_ERROR  7=RECEIVE_ERROR  8=INTERRUPTED  9=UNEXPECTED_ERROR
"""
import argparse
import math
import os
import sys
import time
import traceback

EXIT = {'LOCATED': 0, 'BEARING_ONLY': 13, 'NO_PERSON': 10, 'NO_IMAGE': 3, 'NO_ODOM': 14,
        'NO_CAMINFO': 15, 'CONVERT_ERROR': 4, 'MODEL_ERROR': 5, 'INIT_ERROR': 6,
        'RECEIVE_ERROR': 7, 'INTERRUPTED': 8, 'UNEXPECTED_ERROR': 9}
PERSON_CLASS = 0     # STEP 11 E で確認（names に 'person' は ID 0 の1つだけ）
CAM_X = 0.08         # m  base_footprint から見たカメラの前後位置（STEP 13 で実測 0.080）
CAM_Z = 1.04         # m  床からのカメラの高さ（STEP 13 で実測 1.040）
EDGE_PX = 2.0        # 画像の端から何ピクセル以内を「端に接している」とみなすか（暫定）
MIN_BELOW_HORIZON = 5.0  # y2 - cy がこれ未満なら距離 A を出さない（暫定）


def finish(state, detail=''):
    print(f'結果: {state}')
    if detail:
        print(f'詳細: {detail}')
    sys.stdout.flush()
    sys.exit(EXIT[state])


def check_env():
    """ultralytics を import する前に、安全のための指定がそろっているか確認する。"""
    problems = []
    if not os.environ.get('YOLO_CONFIG_DIR'):
        problems.append('YOLO_CONFIG_DIR が未設定')
    if os.environ.get('YOLO_OFFLINE', '').lower() != 'true':
        problems.append('YOLO_OFFLINE=true ではない')
    if os.environ.get('YOLO_AUTOINSTALL', '').lower() != 'false':
        problems.append('YOLO_AUTOINSTALL=false ではない')
    if not sys.dont_write_bytecode:
        problems.append('PYTHONDONTWRITEBYTECODE=1 / python3 -B ではない')
    return problems


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_sec(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def receive_all(image_topic, info_topic, odom_topic, timeout):
    """(状態, {image, info, odom}, 詳細) を返す。ROS 2 の後始末は、成功・失敗にかかわらずここで行う。"""
    rclpy = None
    initialized = False
    node = None
    got = {'image': None, 'info': None, 'odom': None}
    try:
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from rclpy.wait_for_message import wait_for_message
            from sensor_msgs.msg import CameraInfo, Image
            from nav_msgs.msg import Odometry
            rclpy.init(args=['locate_once', '--ros-args', '--disable-external-lib-logs'])
            initialized = True
            node = rclpy.create_node('locate_once')
        except Exception as e:  # noqa: BLE001
            return 'INIT_ERROR', got, f'ROS 2 の初期化に失敗: {type(e).__name__}: {e}'

        for key, msg_type, topic, fail_state in (('info', CameraInfo, info_topic, 'NO_CAMINFO'),
                                                 ('image', Image, image_topic, 'NO_IMAGE'),
                                                 ('odom', Odometry, odom_topic, 'NO_ODOM')):
            ok, msg = False, None
            try:
                ok, msg = wait_for_message(msg_type, node, topic,
                                           qos_profile=qos_profile_sensor_data, time_to_wait=timeout)
            except Exception as e:  # noqa: BLE001
                return 'RECEIVE_ERROR', got, f'{topic}: {type(e).__name__}: {e}'
            if not ok or msg is None:
                return fail_state, got, f'{topic} を {timeout} 秒以内に受信できない'
            got[key] = msg
        return 'OK', got, ''
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception as e:  # noqa: BLE001
                print(f'[WARN] destroy_node に失敗: {type(e).__name__}: {e}')
        if initialized:
            try:
                rclpy.shutdown()
            except Exception as e:  # noqa: BLE001
                print(f'[WARN] rclpy.shutdown に失敗: {type(e).__name__}: {e}')


def main():
    p = argparse.ArgumentParser(description='人物のワールド座標を1回だけ推定する')
    p.add_argument('--model', default='/home/ubuntu/models/yolov8n.pt')
    p.add_argument('--image-topic', default='/camera/image_raw')
    p.add_argument('--info-topic', default='/camera/camera_info')
    p.add_argument('--odom-topic', default='/odom')
    p.add_argument('--timeout', type=float, default=10.0)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--height', type=float, default=1.7, help='距離 B 用の身長の仮定 [m]')
    p.add_argument('--gt', type=float, nargs=2, metavar=('X', 'Y'), default=None,
                   help='アクターの基準点のワールド座標（指定したときだけ誤差を計算）')
    args = p.parse_args()
    if args.timeout <= 0:
        p.error('--timeout は 0 より大きい値にすること')
    if not 0.0 < args.conf <= 1.0:
        p.error('--conf は 0 より大きく 1 以下にすること')
    if args.height <= 0:
        p.error('--height は 0 より大きい値にすること')

    problems = check_env()
    if problems:
        finish('INIT_ERROR', '、'.join(problems) + '（ultralytics を import せずに終了）')
    if not os.path.isfile(args.model):
        finish('MODEL_ERROR', f'モデルファイルがない: {args.model}（ダウンロードはしない）')
    try:
        from cv_bridge import CvBridge
    except Exception as e:  # noqa: BLE001
        finish('INIT_ERROR', f'cv_bridge の import に失敗: {type(e).__name__}: {e}')

    state, got, detail = receive_all(args.image_topic, args.info_topic, args.odom_topic, args.timeout)
    if state != 'OK':
        finish(state, detail)
    img_msg, info, odom = got['image'], got['info'], got['odom']

    fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
    w, h = img_msg.width, img_msg.height
    t_img, t_odom = stamp_sec(img_msg.header), stamp_sec(odom.header)
    xr, yr = odom.pose.pose.position.x, odom.pose.pose.position.y
    psi = yaw_from_quat(odom.pose.pose.orientation)
    print(f'camera_info: fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f} size={info.width}x{info.height}'
          f'（今回受信した値を使用）')
    print(f'画像: {w}x{h} encoding={img_msg.encoding} frame_id={img_msg.header.frame_id} stamp={t_img:.3f}')
    print(f'odom: x={xr:.4f} y={yr:.4f} psi={psi:.4f} rad ({math.degrees(psi):.2f}°) frame_id={odom.header.frame_id} '
          f'stamp={t_odom:.3f}')
    print(f'時刻差（画像 - odom）: {t_img - t_odom:+.3f} s（ロボットが動いていればこの差の分だけ位置がずれる）')
    print(f'カメラの取り付け（前提）: base_footprint から ({CAM_X}, 0, {CAM_Z}) m、回転なし')

    try:
        img = CvBridge().imgmsg_to_cv2(img_msg, 'bgr8')
    except Exception as e:  # noqa: BLE001
        finish('CONVERT_ERROR', f'{type(e).__name__}: {e}')

    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
        t1 = time.monotonic()
        results = model.predict(img, classes=[PERSON_CLASS], conf=args.conf,
                                device='cpu', save=False, verbose=False)
        infer_s = time.monotonic() - t1
        boxes = results[0].boxes
        dets = [(float(c), [float(v) for v in b])
                for c, b in zip(boxes.conf.tolist(), boxes.xyxy.tolist())]
    except Exception as e:  # noqa: BLE001
        finish('MODEL_ERROR', f'{type(e).__name__}: {e}')

    print(f'推論時間: {infer_s:.3f}s（CPU、初回はモデルの準備時間を含む）')
    for i, (c, b) in enumerate(dets):
        print(f'  person[{i}] conf={c:.3f} xyxy=[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]')
    if not dets:
        finish('NO_PERSON', f'conf閾値={args.conf}（推論は成功し、閾値以上の人物は検出されなかった）')

    conf, (x1, y1, x2, y2) = max(dets, key=lambda d: d[0])
    print(f'使う検出: 信頼度が最大の person（conf={conf:.3f}）')
    flags = []
    if y2 >= h - 1 - EDGE_PX:
        flags.append('BOTTOM_CUT')
    if y1 <= EDGE_PX:
        flags.append('TOP_CUT')
    if x1 <= EDGE_PX or x2 >= w - 1 - EDGE_PX:
        flags.append('SIDE_CUT')
    if 'BOTTOM_CUT' not in flags and (y2 - cy) < MIN_BELOW_HORIZON:
        flags.append('ABOVE_HORIZON')
    print(f'枠の状態: {", ".join(flags) if flags else "端に接していない"}')

    u_c = 0.5 * (x1 + x2)
    phi = math.atan2(cx - u_c, fx)
    side = '左' if phi > 0 else ('右' if phi < 0 else '正面')
    note = '（SIDE_CUT のため参考値）' if 'SIDE_CUT' in flags else ''
    print(f'方向: u_c={u_c:.1f} → phi={math.degrees(phi):+.2f}°（{side}）{note}')

    z_a = None
    if 'BOTTOM_CUT' in flags or 'ABOVE_HORIZON' in flags:
        print('距離 A（足元）: 確定しない（' + ', '.join(f for f in flags if f in ('BOTTOM_CUT', 'ABOVE_HORIZON')) + '）')
    else:
        z_a = fy * CAM_Z / (y2 - cy)
        print(f'距離 A（足元、前提：カメラ水平・床が平坦）: Z={z_a:.3f} m（カメラの光軸方向）')
    if 'BOTTOM_CUT' in flags or 'TOP_CUT' in flags:
        print('距離 B（枠の高さ、参考値）: 無効（枠が画像の上下端で切れている）')
    else:
        z_b = fy * args.height / (y2 - y1)
        print(f'距離 B（枠の高さ、身長 {args.height} m の仮定による参考値）: Z={z_b:.3f} m')

    if z_a is None:
        finish('BEARING_ONLY', '人物は検出したが、足元からの距離を確定できない。座標は出さない')

    px = CAM_X + z_a
    py = z_a * (cx - u_c) / fx
    X = xr + px * math.cos(psi) - py * math.sin(psi)
    Y = yr + px * math.sin(psi) + py * math.cos(psi)
    print(f'ロボット座標: px={px:.3f} m（前）, py={py:+.3f} m（左が正）')
    print(f'推定ワールド座標: X={X:.3f}, Y={Y:.3f}')
    if args.gt is not None:
        gx, gy = args.gt
        dx, dy = X - gx, Y - gy
        print(f'真値（アクターの基準点）: ({gx}, {gy})。基準点は足元の接地点と同じとは限らない')
        print(f'誤差: dX={dx:+.3f} m, dY={dy:+.3f} m, 平面誤差={math.hypot(dx, dy):.3f} m')
    finish('LOCATED', f'X={X:.3f} Y={Y:.3f} conf={conf:.3f}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        finish('INTERRUPTED', 'KeyboardInterrupt')
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        finish('UNEXPECTED_ERROR', '想定外の例外（上の traceback を参照）')
