#!/usr/bin/env python3
"""save_frame.py : /camera/image_raw を1枚だけ受信し、PNG で保存する（確認用）。

しないこと：/cmd_vel への送信、推論、画面表示。保存するのは指定した PNG 1枚だけ。
既存のファイルは上書きしない。ROS 2 の終了処理は、受信の成否にかかわらず変換・保存の前に済ませる。
終了コード: 0=SAVED, 2=引数の誤り(argparse), 3=NO_IMAGE（時間切れ）, 4=CONVERT_ERROR, 6=INIT_ERROR,
            7=RECEIVE_ERROR, 9=UNEXPECTED_ERROR, 11=EXISTS, 12=WRITE_ERROR
"""
import argparse
import os
import sys
import traceback

EXIT = {'SAVED': 0, 'NO_IMAGE': 3, 'CONVERT_ERROR': 4, 'INIT_ERROR': 6, 'RECEIVE_ERROR': 7,
        'UNEXPECTED_ERROR': 9, 'EXISTS': 11, 'WRITE_ERROR': 12}


def finish(state, detail=''):
    print(f'結果: {state}')
    if detail:
        print(f'詳細: {detail}')
    sys.stdout.flush()
    sys.exit(EXIT[state])


def receive_one(topic, timeout):
    """(状態, メッセージ, 詳細) を返す。ROS 2 の後始末は、成功・失敗にかかわらずここで行う。"""
    rclpy = None
    initialized = False
    node = None
    try:
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from rclpy.wait_for_message import wait_for_message
            from sensor_msgs.msg import Image
            rclpy.init(args=['save_frame', '--ros-args', '--disable-external-lib-logs'])
            initialized = True
            node = rclpy.create_node('save_frame')
        except Exception as e:  # noqa: BLE001
            return 'INIT_ERROR', None, f'ROS 2 の初期化に失敗: {type(e).__name__}: {e}'

        ok, msg = False, None
        try:
            ok, msg = wait_for_message(Image, node, topic,
                                       qos_profile=qos_profile_sensor_data, time_to_wait=timeout)
        except Exception as e:  # noqa: BLE001
            return 'RECEIVE_ERROR', None, f'{type(e).__name__}: {e}'
        if not ok or msg is None:
            return 'NO_IMAGE', None, f'{topic} を {timeout} 秒以内に受信できない'
        return 'OK', msg, ''
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
    p = argparse.ArgumentParser(description='カメラ画像を1枚だけ PNG で保存する')
    p.add_argument('--out', required=True)
    p.add_argument('--topic', default='/camera/image_raw')
    p.add_argument('--timeout', type=float, default=10.0)
    args = p.parse_args()
    if args.timeout <= 0:
        p.error('--timeout は 0 より大きい値にすること')
    if os.path.exists(args.out):
        finish('EXISTS', f'既に存在するので上書きしない: {args.out}')
    try:
        import cv2
        from cv_bridge import CvBridge
    except Exception as e:  # noqa: BLE001
        finish('INIT_ERROR', f'cv2 / cv_bridge の import に失敗: {type(e).__name__}: {e}')

    state, msg, detail = receive_one(args.topic, args.timeout)  # ここで ROS 2 は終了済み
    if state != 'OK':
        finish(state, detail)
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    print(f'画像: {msg.width}x{msg.height} encoding={msg.encoding} frame_id={msg.header.frame_id} stamp={stamp:.3f}')

    try:
        img = CvBridge().imgmsg_to_cv2(msg, 'bgr8')
    except Exception as e:  # noqa: BLE001
        finish('CONVERT_ERROR', f'{type(e).__name__}: {e}')
    if os.path.exists(args.out):
        finish('EXISTS', f'受信中に作成されたので上書きしない: {args.out}')
    try:
        written = cv2.imwrite(args.out, img)
    except Exception as e:  # noqa: BLE001
        finish('WRITE_ERROR', f'{type(e).__name__}: {e}')
    if not written:
        finish('WRITE_ERROR', f'書き込めない: {args.out}')
    finish('SAVED', args.out)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        finish('UNEXPECTED_ERROR', '想定外の例外（上の traceback を参照）')
