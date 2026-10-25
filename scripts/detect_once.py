#!/usr/bin/env python3
"""detect_once.py : /camera/image_raw を1枚だけ受信し、YOLOv8n で人物（class 0）を1回だけ推論して表示する。

しないこと：/cmd_vel への送信、画像の保存、画面表示、モデルのダウンロード、パッケージのインストール。
ただし、ライブラリ（ROS 2・DDS・ultralytics・matplotlib など）によるファイルの書き込みを完全に防げるとは限らない。

実行時に必要な指定（そろっていなければ INIT_ERROR で終了し、ultralytics を import しない）:
  YOLO_CONFIG_DIR=<設定の保存先>  YOLO_OFFLINE=true  YOLO_AUTOINSTALL=false
  PYTHONDONTWRITEBYTECODE=1 または python3 -B

結果は最後の「結果: …」の行で判断する。終了コード:
  0=PERSON_FOUND
  10=NO_PERSON（推論は成功し、人物は検出されなかった。カメラやモデル全体の正しさの証明ではない）
  2=引数の誤り(argparse)  3=NO_IMAGE（時間切れ）  4=CONVERT_ERROR  5=MODEL_ERROR
  6=INIT_ERROR  7=RECEIVE_ERROR（受信処理の例外）  8=INTERRUPTED  9=UNEXPECTED_ERROR
"""
import argparse
import os
import sys
import time
import traceback

EXIT = {'PERSON_FOUND': 0, 'NO_PERSON': 10, 'NO_IMAGE': 3, 'CONVERT_ERROR': 4, 'MODEL_ERROR': 5,
        'INIT_ERROR': 6, 'RECEIVE_ERROR': 7, 'INTERRUPTED': 8, 'UNEXPECTED_ERROR': 9}
PERSON_CLASS = 0  # STEP 11 E で確認（names に 'person' は ID 0 の1つだけ）


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


def receive_one(topic, timeout):
    """(状態, メッセージ, 待ち時間[s], 詳細) を返す。ROS 2 の後始末は、成功・失敗にかかわらずここで行う。"""
    rclpy = None
    initialized = False
    node = None
    try:
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from rclpy.wait_for_message import wait_for_message
            from sensor_msgs.msg import Image
            rclpy.init(args=['detect_once', '--ros-args', '--disable-external-lib-logs'])
            initialized = True
            node = rclpy.create_node('detect_once')
        except Exception as e:  # noqa: BLE001
            return 'INIT_ERROR', None, -1.0, f'ROS 2 の初期化に失敗: {type(e).__name__}: {e}'

        t0 = time.monotonic()
        try:
            ok, msg = wait_for_message(Image, node, topic,
                                       qos_profile=qos_profile_sensor_data, time_to_wait=timeout)
        except Exception as e:  # noqa: BLE001
            return 'RECEIVE_ERROR', None, time.monotonic() - t0, f'{type(e).__name__}: {e}'
        wait_s = time.monotonic() - t0
        if not ok or msg is None:
            return 'NO_IMAGE', None, wait_s, f'{topic} を {timeout} 秒以内に受信できない'
        return 'OK', msg, wait_s, ''
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
    p = argparse.ArgumentParser(description='カメラ画像1枚で人物検出を1回だけ行う')
    p.add_argument('--model', default='/home/ubuntu/models/yolov8n.pt')
    p.add_argument('--topic', default='/camera/image_raw')
    p.add_argument('--timeout', type=float, default=10.0)
    p.add_argument('--conf', type=float, default=0.25)
    args = p.parse_args()
    if args.timeout <= 0:
        p.error('--timeout は 0 より大きい値にすること')
    if not 0.0 < args.conf <= 1.0:
        p.error('--conf は 0 より大きく 1 以下にすること')

    problems = check_env()
    if problems:
        finish('INIT_ERROR', '、'.join(problems) + '（ultralytics を import せずに終了）')
    if not os.path.isfile(args.model):
        finish('MODEL_ERROR', f'モデルファイルがない: {args.model}（ダウンロードはしない）')
    try:
        from cv_bridge import CvBridge
    except Exception as e:  # noqa: BLE001
        finish('INIT_ERROR', f'cv_bridge の import に失敗: {type(e).__name__}: {e}')

    state, msg, wait_s, detail = receive_one(args.topic, args.timeout)
    if state != 'OK':
        finish(state, detail)
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    print(f'画像: {msg.width}x{msg.height} encoding={msg.encoding} frame_id={msg.header.frame_id} '
          f'stamp={stamp:.3f} 受信までの時間={wait_s:.2f}s')

    try:
        img = CvBridge().imgmsg_to_cv2(msg, 'bgr8')
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
        dets = [(float(c), [round(float(v), 1) for v in b])
                for c, b in zip(boxes.conf.tolist(), boxes.xyxy.tolist())]
    except Exception as e:  # noqa: BLE001
        finish('MODEL_ERROR', f'{type(e).__name__}: {e}')

    print(f'推論時間: {infer_s:.3f}s（CPU、初回はモデルの準備時間を含む）')
    for i, (c, b) in enumerate(dets):
        print(f'  person[{i}] conf={c:.3f} xyxy={b}')
    if dets:
        finish('PERSON_FOUND', f'人数={len(dets)} conf閾値={args.conf}')
    finish('NO_PERSON', f'人数=0 conf閾値={args.conf}'
                        '（推論は成功し、人物は検出されなかった。カメラやモデル全体の正しさの証明ではない）')


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
