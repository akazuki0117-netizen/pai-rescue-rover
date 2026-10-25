#!/usr/bin/env python3
"""detect_image.py : 既存の画像ファイル1枚を YOLOv8n で推論し、人物（class 0）の検出結果を表示する。

検出器そのものの確認用（ROS 2・Gazebo は使わない）。
しないこと：画像・結果の保存、画面表示、モデルのダウンロード、パッケージのインストール。
ただし、ライブラリ（ultralytics・torch・matplotlib など）によるファイルの書き込みを完全に防げるとは限らない。

実行時に必要な指定（そろっていなければ INIT_ERROR で終了し、ultralytics を import しない）:
  YOLO_CONFIG_DIR=<設定の保存先>  YOLO_OFFLINE=true  YOLO_AUTOINSTALL=false
  PYTHONDONTWRITEBYTECODE=1 または python3 -B

結果は最後の「結果: …」の行で判断する。終了コード:
  0=PERSON_FOUND  10=NO_PERSON（推論は成功し、閾値以上の人物は検出されなかった）
  2=引数の誤り(argparse)  4=IMAGE_ERROR  5=MODEL_ERROR  6=INIT_ERROR  8=INTERRUPTED  9=UNEXPECTED_ERROR
"""
import argparse
import os
import sys
import time
import traceback

EXIT = {'PERSON_FOUND': 0, 'NO_PERSON': 10, 'IMAGE_ERROR': 4, 'MODEL_ERROR': 5,
        'INIT_ERROR': 6, 'INTERRUPTED': 8, 'UNEXPECTED_ERROR': 9}
PERSON_CLASS = 0  # STEP 11 E で確認（names に 'person' は ID 0 の1つだけ）
DEFAULT_IMAGE = '/home/ubuntu/.local/lib/python3.10/site-packages/ultralytics/assets/bus.jpg'


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


def main():
    p = argparse.ArgumentParser(description='画像ファイル1枚で人物検出を1回だけ行う')
    p.add_argument('--image', default=DEFAULT_IMAGE)
    p.add_argument('--model', default='/home/ubuntu/models/yolov8n.pt')
    p.add_argument('--conf', type=float, default=0.25)
    args = p.parse_args()
    if not 0.0 < args.conf <= 1.0:
        p.error('--conf は 0 より大きく 1 以下にすること')

    problems = check_env()
    if problems:
        finish('INIT_ERROR', '、'.join(problems) + '（ultralytics を import せずに終了）')
    if not os.path.isfile(args.model):
        finish('MODEL_ERROR', f'モデルファイルがない: {args.model}（ダウンロードはしない）')
    if not os.path.isfile(args.image):
        finish('IMAGE_ERROR', f'画像ファイルがない: {args.image}')

    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        finish('INIT_ERROR', f'cv2 の import に失敗: {type(e).__name__}: {e}')
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)  # BGR で読み込む（書き込みはしない）
    if img is None:
        finish('IMAGE_ERROR', f'画像を読み込めない: {args.image}')
    h, w = img.shape[:2]
    print(f'画像: {args.image} {w}x{h}')

    try:
        from ultralytics import YOLO
        t0 = time.monotonic()
        model = YOLO(args.model)
        load_s = time.monotonic() - t0
        t1 = time.monotonic()
        results = model.predict(img, classes=[PERSON_CLASS], conf=args.conf,
                                device='cpu', save=False, verbose=False)
        infer_s = time.monotonic() - t1
        boxes = results[0].boxes
        dets = [(float(c), [round(float(v), 1) for v in b])
                for c, b in zip(boxes.conf.tolist(), boxes.xyxy.tolist())]
    except Exception as e:  # noqa: BLE001
        finish('MODEL_ERROR', f'{type(e).__name__}: {e}')

    print(f'モデル読み込み: {load_s:.3f}s / 推論（1回目、ウォームアップを含む可能性あり）: {infer_s:.3f}s（CPU）')
    for i, (c, b) in enumerate(dets):
        print(f'  person[{i}] conf={c:.3f} xyxy={b}')
    if dets:
        finish('PERSON_FOUND', f'人数={len(dets)} conf閾値={args.conf}')
    finish('NO_PERSON', f'人数=0 conf閾値={args.conf}（推論は成功し、閾値以上の人物は検出されなかった）')


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
