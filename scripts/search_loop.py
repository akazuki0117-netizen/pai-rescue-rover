#!/usr/bin/env python3
"""search_loop.py : その場で人物の検出とワールド座標の推定を繰り返し、CSV に記録する（初版）。

しないこと：/cmd_vel の購読・送信、ロボットの移動・旋回、画像の保存、画面表示、モデルのダウンロード、インストール。
ライブラリ（ROS 2・DDS・ultralytics など）によるファイルの書き込みを完全に防げるとは限らない。
計算は person_geometry.py（locate_once.py の式をそのまま移したもの）を使う。

構造: ROS 2 の受信は別スレッドの SingleThreadedExecutor で回し続け、メインスレッドで YOLO を実行する。
      推論中も受信は続く（GIL のため Python の処理がわずかに遅れることはある）。
時刻: 画像と /odom はどちらも header.stamp（シミュレーション時間）で比べる。
      推論の前に時刻差を確かめ、|画像 - odom| > --max-dt なら ODOM_TIME_MISMATCH として推論も座標計算もしない。
クラスタ: LOCATED の座標を半径 --merge-radius でまとめる暫定の位置クラスタ。同じ人物であることは保証しない。

1周の状態: LOCATED / BEARING_ONLY / NO_PERSON / NO_IMAGE / NO_ODOM / ODOM_TIME_MISMATCH / CYCLE_ERROR
終了コード: 0=COMPLETED（LOCATED が1回以上）  10=COMPLETED_NO_LOCATION  2=引数の誤り(argparse)
            5=MODEL_ERROR（読み込み失敗）  6=INIT_ERROR（import・rclpy.init・ノード作成の失敗）
            8=INTERRUPTED  9=UNEXPECTED_ERROR  12=WRITE_ERROR  15=NO_CAMINFO
sys.exit（SystemExit）を出すのは main() の最後と最上位の例外処理だけ。run() は (状態, 詳細) を返す。
受信スレッドが後始末で終了しなかった場合は、結果に「後始末未完了」と明示する（終了コードは変えない）。
実行時に必要な指定: YOLO_CONFIG_DIR / YOLO_OFFLINE=true / YOLO_AUTOINSTALL=false / python3 -B
"""
import argparse
import bisect
import collections
import csv
import math
import os
import statistics
import sys
import threading
import time
import traceback

import person_geometry as pg

EXIT = {'COMPLETED': 0, 'COMPLETED_NO_LOCATION': 10, 'MODEL_ERROR': 5, 'INIT_ERROR': 6,
        'INTERRUPTED': 8, 'UNEXPECTED_ERROR': 9, 'WRITE_ERROR': 12, 'NO_CAMINFO': 15}
PERSON_CLASS = 0
RAW_FIELDS = ['cycle', 'wall_s', 'img_stamp', 'odom_stamp', 'dt_img_minus_odom', 'status', 'flags',
              'n_det', 'conf', 'x1', 'y1', 'x2', 'y2', 'u_c', 'phi_deg', 'z_a', 'z_b',
              'xr', 'yr', 'psi_deg', 'px', 'py', 'X', 'Y', 'cluster_id']
CLUSTER_FIELDS = ['cluster_id', 'n', 'mean_X', 'mean_Y', 'std_X', 'std_Y', 'max_conf', 'merge_radius',
                  'gt_X', 'gt_Y', 'err_X', 'err_Y', 'err_planar']


def finish(state, detail=''):
    print(f'結果: {state}')
    if detail:
        print(f'詳細: {detail}')
    sys.stdout.flush()
    sys.exit(EXIT[state])


def check_env():
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


def stamp_sec(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def fmt(v, nd=4):
    return '' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else v)


class Receiver:
    """ROS 2 の受信だけを担当する。最新の画像、最初の camera_info、直近の odom を保持する。"""

    def __init__(self, node, args):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from nav_msgs.msg import Odometry
        self.lock = threading.Lock()
        self.image = None
        self.image_seq = 0
        self.info = None
        self.odom = collections.deque(maxlen=args.odom_buffer)  # (t, x, y, psi)
        node.create_subscription(Image, args.image_topic, self._on_image, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, args.info_topic, self._on_info, qos_profile_sensor_data)
        node.create_subscription(Odometry, args.odom_topic, self._on_odom, qos_profile_sensor_data)

    def _on_image(self, msg):
        with self.lock:
            self.image = msg
            self.image_seq += 1

    def _on_info(self, msg):
        with self.lock:
            if self.info is None:
                self.info = msg

    def _on_odom(self, msg):
        p = msg.pose.pose
        with self.lock:
            self.odom.append((stamp_sec(msg.header), p.position.x, p.position.y,
                              pg.yaw_from_quat(p.orientation)))

    def get_info(self):
        with self.lock:
            return self.info

    def latest_image(self):
        with self.lock:
            return self.image, self.image_seq

    def nearest_odom(self, t):
        with self.lock:
            buf = list(self.odom)
        if not buf:
            return None
        times = [o[0] for o in buf]
        i = bisect.bisect_left(times, t)
        cands = [buf[j] for j in (i - 1, i) if 0 <= j < len(buf)]
        return min(cands, key=lambda o: abs(o[0] - t))


class PositionClusters:
    """LOCATED の座標を、既存クラスタの平均から半径 r 以内ならまとめる暫定の位置クラスタ。
    同じ人物であることは保証しない（追跡・データ関連付けはしない）。"""

    def __init__(self, radius):
        self.radius = radius
        self.clusters = []  # dict(id, xs, ys, confs)

    def add(self, X, Y, conf):
        best, best_d = None, None
        for c in self.clusters:
            d = math.hypot(X - statistics.fmean(c['xs']), Y - statistics.fmean(c['ys']))
            if d <= self.radius and (best_d is None or d < best_d):
                best, best_d = c, d
        if best is None:
            best = {'id': len(self.clusters) + 1, 'xs': [], 'ys': [], 'confs': []}
            self.clusters.append(best)
        best['xs'].append(X)
        best['ys'].append(Y)
        best['confs'].append(conf)
        return best['id']


def open_new_csv(path, fields):
    f = open(path, 'x', newline='')  # 'x' = 既存ファイルがあれば失敗する（上書きしない）
    try:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        f.flush()
    except BaseException:
        f.close()          # ヘッダーの書き込みに失敗しても、開いたファイルは必ず閉じる
        raise
    return f, w


def one_cycle(cycle, t_start, rx, state, args, model, bridge, clusters, info_k):
    """1周分の処理。記録用の row を返す。例外は呼び出し側で CYCLE_ERROR にする。"""
    fx, fy, cx, cy = info_k
    row = {k: '' for k in RAW_FIELDS}
    row.update(cycle=cycle, wall_s=fmt(time.monotonic() - t_start, 3))

    deadline = time.monotonic() + args.timeout
    img_msg, seq = rx.latest_image()
    while (img_msg is None or seq == state['last_seq']) and time.monotonic() < deadline:
        time.sleep(0.02)
        img_msg, seq = rx.latest_image()
    if img_msg is None or seq == state['last_seq']:
        row['status'] = 'NO_IMAGE'
        return row
    state['last_seq'] = seq
    t_img = stamp_sec(img_msg.header)
    row['img_stamp'] = fmt(t_img, 3)

    od = rx.nearest_odom(t_img)
    if od is None:
        row['status'] = 'NO_ODOM'
        return row
    t_od, xr, yr, psi = od
    dt = t_img - t_od
    row.update(odom_stamp=fmt(t_od, 3), dt_img_minus_odom=fmt(dt, 3),
               xr=fmt(xr), yr=fmt(yr), psi_deg=fmt(math.degrees(psi), 3))
    if abs(dt) > args.max_dt:            # 推論の前に判定。座標も統合も行わない
        row['status'] = 'ODOM_TIME_MISMATCH'
        return row

    img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
    res = model.predict(img, classes=[PERSON_CLASS], conf=args.conf,
                        device='cpu', save=False, verbose=False)
    b = res[0].boxes
    dets = [(float(c), [float(v) for v in bb]) for c, bb in zip(b.conf.tolist(), b.xyxy.tolist())]
    row['n_det'] = len(dets)
    if not dets:
        row['status'] = 'NO_PERSON'
        return row

    conf, box = max(dets, key=lambda d: d[0])
    g = pg.locate(box, img_msg.width, img_msg.height, fx, fy, cx, cy, xr, yr, psi, args.height)
    row.update(conf=fmt(conf, 3), x1=fmt(box[0], 1), y1=fmt(box[1], 1), x2=fmt(box[2], 1),
               y2=fmt(box[3], 1), u_c=fmt(g['u_c'], 1), phi_deg=fmt(math.degrees(g['phi']), 3),
               z_a=fmt(g['z_a']), z_b=fmt(g['z_b']), flags='|'.join(g['flags']))
    if g['X'] is None:
        row['status'] = 'BEARING_ONLY'
        return row
    cid = clusters.add(g['X'], g['Y'], conf)
    row.update(status='LOCATED', px=fmt(g['px']), py=fmt(g['py']), X=fmt(g['X']), Y=fmt(g['Y']),
               cluster_id=cid)
    return row


def write_clusters(path, clusters, args):
    f, w = open_new_csv(path, CLUSTER_FIELDS)
    try:
        for c in clusters.clusters:
            n = len(c['xs'])
            mx, my = statistics.fmean(c['xs']), statistics.fmean(c['ys'])
            sx = statistics.stdev(c['xs']) if n > 1 else 0.0
            sy = statistics.stdev(c['ys']) if n > 1 else 0.0
            r = {'cluster_id': c['id'], 'n': n, 'mean_X': fmt(mx), 'mean_Y': fmt(my), 'std_X': fmt(sx),
                 'std_Y': fmt(sy), 'max_conf': fmt(max(c['confs']), 3), 'merge_radius': args.merge_radius}
            msg = (f'cluster {c["id"]}: n={n} mean=({mx:.3f}, {my:.3f}) std=({sx:.3f}, {sy:.3f}) '
                   f'max_conf={max(c["confs"]):.3f}')
            if args.gt is not None:
                ex, ey = mx - args.gt[0], my - args.gt[1]
                r.update(gt_X=args.gt[0], gt_Y=args.gt[1], err_X=fmt(ex), err_Y=fmt(ey),
                         err_planar=fmt(math.hypot(ex, ey)))
                msg += f' 真値との誤差=({ex:+.3f}, {ey:+.3f}) 平面={math.hypot(ex, ey):.3f}'
            w.writerow(r)
            print(msg)
    finally:
        f.close()


def run(args, cleanup):
    """(状態, 詳細) を返す。sys.exit は呼ばない。後始末は finally で必ず試み、結果を cleanup に残す。"""
    try:
        import numpy as np
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.executors import SingleThreadedExecutor
    except Exception as e:  # noqa: BLE001
        return 'INIT_ERROR', f'import に失敗: {type(e).__name__}: {e}'

    initialized = False
    node = executor = spin_thread = raw_f = None
    try:
        try:
            rclpy.init(args=['search_loop', '--ros-args', '--disable-external-lib-logs'])
            initialized = True
            node = rclpy.create_node('search_loop')
        except Exception as e:  # noqa: BLE001
            return 'INIT_ERROR', f'ROS 2 の初期化に失敗: {type(e).__name__}: {e}'
        rx = Receiver(node, args)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        t0 = time.monotonic()
        while rx.get_info() is None and time.monotonic() - t0 < args.timeout:
            time.sleep(0.05)
        info = rx.get_info()
        if info is None:
            return 'NO_CAMINFO', f'{args.info_topic} を {args.timeout} 秒以内に受信できない'
        info_k = (info.k[0], info.k[4], info.k[2], info.k[5])
        print(f'camera_info: fx={info_k[0]:.3f} fy={info_k[1]:.3f} cx={info_k[2]:.3f} cy={info_k[3]:.3f} '
              f'size={info.width}x{info.height}')

        try:
            import torch
            from ultralytics import YOLO
            torch.set_num_threads(args.threads)
            model = YOLO(args.model)
            tw = time.monotonic()
            model.predict(np.zeros((info.height, info.width, 3), dtype=np.uint8), classes=[PERSON_CLASS],
                          conf=args.conf, device='cpu', save=False, verbose=False)
            print(f'YOLO 読み込みとウォームアップ: {time.monotonic() - tw:.3f}s（torch スレッド数 {args.threads}）')
        except Exception as e:  # noqa: BLE001
            return 'MODEL_ERROR', f'{type(e).__name__}: {e}'

        tag = time.strftime('%Y%m%d_%H%M%S')
        raw_path = os.path.join(args.out_dir, f'search_{tag}.csv')
        cl_path = os.path.join(args.out_dir, f'search_{tag}_clusters.csv')
        try:
            os.makedirs(args.out_dir, exist_ok=True)
            raw_f, raw_w = open_new_csv(raw_path, RAW_FIELDS)
        except Exception as e:  # noqa: BLE001
            return 'WRITE_ERROR', f'{raw_path}: {type(e).__name__}: {e}'
        print(f'記録（1周ごと・生の推定値）: {raw_path}')

        bridge = CvBridge()
        clusters = PositionClusters(args.merge_radius)
        counts = collections.Counter()
        state = {'last_seq': 0}
        t_start = time.monotonic()
        for cycle in range(1, args.cycles + 1):
            if time.monotonic() - t_start > args.duration:
                print(f'--duration {args.duration}s に達したので終了')
                break
            t_cycle = time.monotonic()
            try:
                row = one_cycle(cycle, t_start, rx, state, args, model, bridge, clusters, info_k)
            except Exception as e:  # noqa: BLE001
                row = {k: '' for k in RAW_FIELDS}
                row.update(cycle=cycle, wall_s=fmt(t_cycle - t_start, 3), status='CYCLE_ERROR',
                           flags=f'{type(e).__name__}: {e}'[:200])
            counts[row['status']] += 1
            try:
                raw_w.writerow(row)
                raw_f.flush()
            except Exception as e:  # noqa: BLE001
                return 'WRITE_ERROR', f'{raw_path}（{cycle} 周目）: {type(e).__name__}: {e}'
            print(f"[{cycle:3d}] {row['status']:<18} conf={row['conf'] or '-':<6} "
                  f"X={row['X'] or '-':<8} Y={row['Y'] or '-':<8} dt={row['dt_img_minus_odom'] or '-'} "
                  f"flags={row['flags'] or '-'}")
            rest = args.period - (time.monotonic() - t_cycle)
            if rest > 0:
                time.sleep(rest)

        print('=' * 60)
        print('状態ごとの回数: ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())))
        print(f'位置クラスタ（半径 {args.merge_radius} m の暫定。同じ人物であることは保証しない）:')
        try:
            write_clusters(cl_path, clusters, args)
        except Exception as e:  # noqa: BLE001
            return 'WRITE_ERROR', f'{cl_path}: {type(e).__name__}: {e}'
        print(f'クラスタ数: {len(clusters.clusters)} / 記録: {cl_path}')
        if args.gt is not None:
            print('真値はアクターの基準点。足元の接地点と同じとは限らない')
        state_name = 'COMPLETED' if counts['LOCATED'] > 0 else 'COMPLETED_NO_LOCATION'
        return state_name, f'周={sum(counts.values())} LOCATED={counts["LOCATED"]}'
    finally:
        # 後始末の各段階は個別に try で囲み、途中で例外が起きても次の段階へ進む
        if raw_f is not None:
            try:
                raw_f.close()
            except Exception as e:  # noqa: BLE001
                cleanup['problems'].append(f'CSV のクローズに失敗: {type(e).__name__}: {e}')
        if executor is not None:
            try:
                executor.shutdown()
            except Exception as e:  # noqa: BLE001
                cleanup['problems'].append(f'executor.shutdown に失敗: {type(e).__name__}: {e}')
        if spin_thread is not None:
            try:
                spin_thread.join(timeout=2.0)
                if spin_thread.is_alive():
                    cleanup['problems'].append('受信スレッドが 2 秒以内に終了していない（daemon のためプロセス終了時に打ち切られる）')
            except Exception as e:  # noqa: BLE001
                cleanup['problems'].append(f'受信スレッドの join に失敗: {type(e).__name__}: {e}')
        if node is not None:
            try:
                node.destroy_node()
            except Exception as e:  # noqa: BLE001
                cleanup['problems'].append(f'destroy_node に失敗: {type(e).__name__}: {e}')
        if initialized:
            try:
                rclpy.shutdown()
            except Exception as e:  # noqa: BLE001
                cleanup['problems'].append(f'rclpy.shutdown に失敗: {type(e).__name__}: {e}')
        for msg in cleanup['problems']:
            print(f'[WARN] 後始末: {msg}')


def main(cleanup):
    p = argparse.ArgumentParser(description='人物の検出と座標推定を繰り返して記録する（初版）')
    p.add_argument('--model', default='/home/ubuntu/models/yolov8n.pt')
    p.add_argument('--image-topic', default='/camera/image_raw')
    p.add_argument('--info-topic', default='/camera/camera_info')
    p.add_argument('--odom-topic', default='/odom')
    p.add_argument('--cycles', type=int, default=20)
    p.add_argument('--duration', type=float, default=120.0, help='壁時計での上限 [s]')
    p.add_argument('--period', type=float, default=1.0, help='1周の目安 [s]')
    p.add_argument('--timeout', type=float, default=10.0, help='camera_info・画像の待ち時間 [s]')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--height', type=float, default=1.7)
    p.add_argument('--max-dt', type=float, default=0.1, help='画像と odom の時刻差の上限 [s]（暫定）')
    p.add_argument('--merge-radius', type=float, default=0.5, help='位置クラスタの半径 [m]（暫定）')
    p.add_argument('--threads', type=int, default=4, help='torch のスレッド数')
    p.add_argument('--odom-buffer', type=int, default=200)
    p.add_argument('--out-dir', default='/home/ubuntu/runs')
    p.add_argument('--gt', type=float, nargs=2, metavar=('X', 'Y'), default=None)
    args = p.parse_args()
    for name in ('cycles', 'duration', 'period', 'timeout', 'height', 'max_dt', 'merge_radius',
                 'threads', 'odom_buffer'):
        if getattr(args, name) <= 0:
            p.error(f'--{name.replace("_", "-")} は 0 より大きい値にすること')
    if not 0.0 < args.conf <= 1.0:
        p.error('--conf は 0 より大きく 1 以下にすること')

    problems = check_env()
    if problems:
        finish('INIT_ERROR', '、'.join(problems) + '（ultralytics を import せずに終了）')
    if not os.path.isfile(args.model):
        finish('MODEL_ERROR', f'モデルファイルがない: {args.model}（ダウンロードはしない）')
    state, detail = run(args, cleanup)
    if cleanup['problems']:
        detail += f' / 後始末未完了: {len(cleanup["problems"])} 件（上の [WARN] を参照）'
    finish(state, detail)


if __name__ == '__main__':
    _cleanup = {'problems': []}
    try:
        main(_cleanup)
    except KeyboardInterrupt:
        note = '後始末未完了あり（上の [WARN] を参照）' if _cleanup['problems'] else 'run() の finally で後始末を実施'
        finish('INTERRUPTED', f'KeyboardInterrupt（{note}）')
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        finish('UNEXPECTED_ERROR', '想定外の例外（上の traceback を参照）')
