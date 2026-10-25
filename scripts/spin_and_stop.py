#!/usr/bin/env python3
"""spin_and_stop.py : その場旋回 → 停止 → 停止確認（初回テスト用）

安全に関する前提
- 駆動プラグインは最後の指令を保持し、時間切れで止まらない（公開ソース調査。実環境は未確認）。
- このスクリプトが速度0を送れない終わり方（SIGKILL、Gazebo の強制終了、コンテナ停止、
  ターミナルを閉じる等）では、ロボットが回り続ける可能性がある。安全に止まる保証はない。
- 壁との距離は安全の根拠に使っていない。
- publish() が例外なく戻っても、相手に届いたことは保証されない（受け取り側の接続が切れていれば届かない）。
  停止は VERIFY の観測でだけ判断する。
- 車輪ジョイント名と回転方向の符号は実環境で未確認。

使い方（実行は別途承認が必要）
  python3 spin_and_stop.py --dry-run   # PRECHECK のみ。/cmd_vel の送信部品を作らない
  python3 spin_and_stop.py --spin      # 旋回テスト（--spin を明示したときだけ）
  引数なし・不明な引数・両方指定・重複指定は ARG_ERROR で終了（ROS 2 の初期化も送信もしない）
  ROS 引数は --ros-args の後ろに置く（例: --dry-run --ros-args -r __ns:=/x）

終了コード: 0=DONE / DRYRUN_OK, 1=ABORT（停止は確認済み）, 2=REPORT_FAIL,
            3=PRECHECK_ABORT（送信なし）, 4=PRECHECK_NOT_STILL（送信なし・停止未確認）,
            5=ARG_ERROR（引数の誤り。ROS 2 初期化前に終了・送信なし）
"""
import math
import signal
import sys
import time
import traceback

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

try:  # Humble ではこちらの見込み（未確認）
    from rclpy.qos_event import PublisherEventCallbacks
except ImportError:  # 新しい版
    from rclpy.event_handler import PublisherEventCallbacks

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:
    SignalHandlerOptions = None

# ---------------- 設定（すべて暫定値） ----------------
OMEGA = 0.3                 # rad/s  旋回指令（linear.x は常に 0）
OMEGA_LIMIT = 0.3           # rad/s  コード上の上限
SPIN_SIM_S = 3.0            # s(sim) 旋回時間
SPIN_SIM_LIMIT = 3.0        # s(sim) 旋回時間の上限
RATE_HZ = 20.0              # 指令の送信周期
STOP_WALL_S = 1.0           # s(wall) 速度0を送り続ける時間
VERIFY_ZERO_HZ = 5.0        # VERIFY 中も速度0を送る周期
VERIFY_SIM_TIMEOUT = 3.0
VERIFY_WALL_TIMEOUT = 6.0
PRECHECK_WALL_TIMEOUT = 5.0
PRECHECK_STILL_TIMEOUT = 3.0
SUB_WAIT_WALL = 2.0
SPIN_WALL_LIMIT = 6.0
TOTAL_WALL_LIMIT = 20.0
STALE_WALL = 0.5            # これ以上更新がなければ「古い」
STILL_WINDOW_SIM = 0.5      # 静止条件を続けて満たす時間(sim)
STILL_WHEEL = 0.05          # rad/s  V1
STILL_WZ = 0.01             # rad/s  V2
STILL_DPSI = 0.005          # rad    V3
STILL_DPOS = 0.005          # m      V4
ABORT_POS = 0.05            # m      A4
ABORT_OVERROT = 1.05        # rad    A5
ABORT_REVERSE = -0.05       # rad    A6
SAME_SIGN_W = 0.2           # rad/s  A7（符号の決まりは未確認。誤検出は中断側に出る）
SAME_SIGN_SIM = 0.3         # s(sim) A7
LEFT_JOINT = 'wheel_left_joint'     # 公開SDFの名前。実環境は未確認
RIGHT_JOINT = 'wheel_right_joint'
SUB_NODE = 'turtlebot3_diff_drive'  # STEP 9-A で確認済み
FINAL_ZERO_COUNT = 5
FINAL_ZERO_INTERVAL = 0.05

EXIT_CODE = {'DONE': 0, 'DRYRUN_OK': 0, 'ABORT': 1, 'REPORT_FAIL': 2,
             'PRECHECK_ABORT': 3, 'PRECHECK_NOT_STILL': 4, 'ARG_ERROR': 5}

assert 0.0 < OMEGA <= OMEGA_LIMIT
assert 0.0 < SPIN_SIM_S <= SPIN_SIM_LIMIT


class Terminated(Exception):
    """SIGTERM を受けたときに投げる"""


class Abort(Exception):
    """SPIN 中の中断（理由つき）"""


def _on_sigterm(signum, frame):
    raise Terminated()


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def advance_tick(tick, period):                                        # [R2]
    """次の送信時刻。遅れていたら取り戻さず、今から1周期後に置き直す（連続送信を防ぐ）。"""
    tick += period
    now = time.monotonic()
    if tick < now:
        tick = now + period
    return tick


class ignore_signals:
    """この区間だけ SIGINT / SIGTERM を無視する"""

    def __enter__(self):
        self.prev = (signal.signal(signal.SIGINT, signal.SIG_IGN),
                     signal.signal(signal.SIGTERM, signal.SIG_IGN))

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self.prev[0])
        signal.signal(signal.SIGTERM, self.prev[1])
        return False


class SpinAndStop(Node):
    def __init__(self):
        super().__init__('spin_and_stop')
        # 受信は BEST_EFFORT：送り側が RELIABLE でも BEST_EFFORT でもつながる
        self.create_subscription(Clock, '/clock', self._on_clock, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self._on_odom, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self._on_js, qos_profile_sensor_data)
        self.t_start = time.monotonic()
        self.sim_now = None
        self.clock_wall = None
        self.odom = None          # (x, y, psi, wz)
        self.odom_wall = None
        self.js = None            # (w_left, w_right)
        self.js_wall = None
        self.js_error = None
        self.pub = None
        self.armed = False        # 旋回指令を送り始めたら True。False の間は一切送らない
        self.qos_incompatible = False
        self.zero_ok = 0
        self.zero_fail = 0
        self.last_send_error = None
        self.pump_error = None
        self.report = {}

    # ---------- 受信 ----------
    def _on_clock(self, msg):
        t = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self.sim_now is None or t > self.sim_now:   # 進んだときだけ更新 → 止まれば「古い」になる
            self.sim_now = t
            self.clock_wall = time.monotonic()

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        self.odom = (p.x, p.y, yaw_from_quat(msg.pose.pose.orientation), msg.twist.twist.angular.z)
        self.odom_wall = time.monotonic()

    def _on_js(self, msg):
        names = list(msg.name)
        if LEFT_JOINT not in names or RIGHT_JOINT not in names:
            self.js_error = f'車輪ジョイントが見つからない: {names}'
            return
        if len(msg.velocity) != len(names):
            self.js_error = f'velocity の長さが name と一致しない: {len(msg.velocity)}/{len(names)}'
            return
        self.js = (msg.velocity[names.index(LEFT_JOINT)], msg.velocity[names.index(RIGHT_JOINT)])
        self.js_wall = time.monotonic()
        self.js_error = None

    def stale(self):
        now = time.monotonic()
        return [name for name, t in (('/clock', self.clock_wall),
                                     ('/odom', self.odom_wall),
                                     ('/joint_states', self.js_wall))
                if t is None or now - t > STALE_WALL]

    def pump_until(self, wall_deadline):
        while True:
            remain = wall_deadline - time.monotonic()
            if remain <= 0.0:
                return
            rclpy.spin_once(self, timeout_sec=min(remain, 0.01))

    def safe_pump_until(self, wall_deadline):                         # [R1]
        """受信処理。例外が出たら記録して壁時計で待つだけにする。成功なら True。"""
        try:
            self.pump_until(wall_deadline)
            return True
        except Exception as e:   # noqa: BLE001
            self.pump_error = repr(e)
            remain = wall_deadline - time.monotonic()
            if remain > 0.0:
                time.sleep(remain)
            return False

    # ---------- 送信 ----------
    def create_cmd_publisher(self):
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         depth=1, durability=DurabilityPolicy.VOLATILE)
        cbs = PublisherEventCallbacks(incompatible_qos=self._on_incompatible)
        self.pub = self.create_publisher(Twist, '/cmd_vel', qos, event_callbacks=cbs)

    def _on_incompatible(self, event):
        self.qos_incompatible = True

    def sub_count(self):                                               # [R5]
        """受け取り側の数。Publisher.get_subscription_count が無ければグラフ上の数で代用する。"""
        try:
            return self.pub.get_subscription_count(), 'publisher'
        except AttributeError:
            return self.count_subscribers('/cmd_vel'), 'graph（QoS一致は見ていない）'

    def _send(self, wz):
        """publish が例外なく戻れば True。相手に届いたことの保証ではない。"""
        if self.pub is None:
            return False
        t = Twist()               # linear.* と angular.x/y は常に 0
        t.angular.z = float(wz)
        try:
            self.pub.publish(t)
            return True
        except Exception as e:    # noqa: BLE001
            self.last_send_error = repr(e)
            return False

    def send_zero(self):
        ok = self._send(0.0)
        if ok:
            self.zero_ok += 1
        else:
            self.zero_fail += 1
        return ok

    # ---------- 静止判定（V1〜V5） ----------
    def wait_still(self, sim_timeout, wall_timeout, send_zero_hz=None):
        wall0 = time.monotonic()
        sim0 = self.sim_now
        stable_since, ref = None, None
        next_zero = wall0
        last = {}
        while True:
            now = time.monotonic()
            if send_zero_hz and now >= next_zero:
                self.send_zero()
                next_zero = now + 1.0 / send_zero_hz
            if not self.safe_pump_until(now + 0.02):                   # [R1] 受信不能＝確認不能
                return False, f'受信処理に失敗: {self.pump_error}', last
            if time.monotonic() - wall0 > wall_timeout:
                return False, f'壁時計 {wall_timeout}s 以内に静止を確認できない', last
            if self.sim_now is not None and sim0 is not None and self.sim_now - sim0 > sim_timeout:
                return False, f'シミュ時間 {sim_timeout}s 以内に静止を確認できない', last
            bad = self.stale()
            if bad or self.js_error:
                stable_since = None
                last = {'stale': bad, 'js_error': self.js_error}
                continue
            wl, wr = self.js
            x, y, psi, wz = self.odom
            last = {'w_left': wl, 'w_right': wr, 'wz': wz, 'sim': self.sim_now}
            if abs(wl) >= STILL_WHEEL or abs(wr) >= STILL_WHEEL or abs(wz) >= STILL_WZ:  # V1, V2
                stable_since = None
                continue
            if stable_since is None:
                stable_since, ref = self.sim_now, (x, y, psi)
                continue
            dpos = math.hypot(x - ref[0], y - ref[1])
            dpsi = abs(wrap(psi - ref[2]))
            last.update(dpos=dpos, dpsi=dpsi, still_for=self.sim_now - stable_since)
            if dpos >= STILL_DPOS or dpsi >= STILL_DPSI:                                   # V3, V4
                stable_since, ref = self.sim_now, (x, y, psi)
                continue
            if self.sim_now - stable_since >= STILL_WINDOW_SIM:  # V5：シミュ時間が進んで初めて成立
                return True, 'OK', last


# ================= 状態ごとの処理 =================
def precheck(node, dry_run):
    """(成功か, 理由, 失敗時の状態名) を返す。ここでは /cmd_vel に一度も送らない。"""
    deadline = time.monotonic() + PRECHECK_WALL_TIMEOUT
    first_sim = None
    while time.monotonic() < deadline:
        node.pump_until(time.monotonic() + 0.05)
        if first_sim is None and node.sim_now is not None:
            first_sim = node.sim_now
        if (first_sim is not None and node.sim_now > first_sim
                and not node.stale() and node.js_error is None):
            break
    else:
        return False, (f'受信待ちの時間切れ: 古い/未受信={node.stale()} '
                       f'js_error={node.js_error} sim開始値={first_sim} sim現在={node.sim_now}'), 'PRECHECK_ABORT'

    pubs = node.count_publishers('/cmd_vel')
    if pubs != 0:
        return False, f'/cmd_vel に他の送り側がいる: {pubs}', 'PRECHECK_ABORT'
    subs = node.get_subscriptions_info_by_topic('/cmd_vel')
    if len(subs) != 1:
        return False, f'/cmd_vel の受け取り側が1個ではない: {len(subs)}', 'PRECHECK_ABORT'
    s = subs[0]
    if s.node_name != SUB_NODE or s.qos_profile.reliability != ReliabilityPolicy.RELIABLE:
        return False, (f'受け取り側が想定と違う: node={s.node_name} '
                       f'reliability={s.qos_profile.reliability}'), 'PRECHECK_ABORT'

    still, why, last = node.wait_still(PRECHECK_STILL_TIMEOUT, PRECHECK_STILL_TIMEOUT)
    if not still:
        return False, f'開始前の静止を確認できない: {why} {last}', 'PRECHECK_NOT_STILL'
    if dry_run:
        return True, 'dry-run: 送信部品は作っていない', None

    node.create_cmd_publisher()   # 作るだけで、まだ送らない
    deadline = time.monotonic() + SUB_WAIT_WALL
    while time.monotonic() < deadline and node.sub_count()[0] != 1:
        node.pump_until(time.monotonic() + 0.05)
    node.pump_until(time.monotonic() + 0.2)   # QoS 不一致の通知を待つ余裕
    count, method = node.sub_count()
    node.report['sub_count_method'] = method
    if count != 1 or node.qos_incompatible:
        return False, (f'受け取り側とつながらない: count={count}（{method}） '
                       f'qos_incompatible={node.qos_incompatible}'), 'PRECHECK_ABORT'
    if node.count_publishers('/cmd_vel') != 1:
        return False, '自分以外の送り側が現れた', 'PRECHECK_ABORT'
    return True, 'OK', None


def spin(node):
    """正常に旋回時間を終えれば戻る。異常は Abort を投げる（その他の例外は main で STOP へ）。"""
    x0, y0, psi0, _ = node.odom
    sim0, wall0 = node.sim_now, time.monotonic()
    acc, last_psi = 0.0, psi0
    same_sign_since = None
    max_w = max_pos = 0.0
    period = 1.0 / RATE_HZ
    next_tick = wall0
    node.armed = True
    while True:
        if not node._send(OMEGA):
            raise Abort(f'旋回指令の送信に失敗: {node.last_send_error}')
        next_tick = advance_tick(next_tick, period)                    # [R2]
        node.pump_until(next_tick)
        now = time.monotonic()

        bad = node.stale()                                             # A1〜A3
        if bad:
            raise Abort(f'データが古い: {bad}')
        if node.js_error:
            raise Abort(node.js_error)
        x, y, psi, _ = node.odom
        wl, wr = node.js
        acc += wrap(psi - last_psi)
        last_psi = psi
        pos = math.hypot(x - x0, y - y0)
        max_w, max_pos = max(max_w, abs(wl), abs(wr)), max(max_pos, pos)
        node.report.update(dpsi=acc, max_wheel=max_w, max_pos=max_pos,
                           spin_sim=node.sim_now - sim0, spin_wall=now - wall0)

        if pos > ABORT_POS:                                            # A4
            raise Abort(f'位置が動いた: {pos:.3f} m')
        if acc > ABORT_OVERROT:                                        # A5
            raise Abort(f'回りすぎ: {acc:.3f} rad')
        if acc < ABORT_REVERSE:                                        # A6
            raise Abort(f'逆向きに回っている: {acc:.3f} rad')
        if abs(wl) > SAME_SIGN_W and abs(wr) > SAME_SIGN_W and (wl > 0) == (wr > 0):  # A7
            if same_sign_since is None:
                same_sign_since = node.sim_now
            if node.sim_now - same_sign_since >= SAME_SIGN_SIM:
                raise Abort(f'左右の車輪が同じ向きに回っている: {wl:.2f}, {wr:.2f}')
        else:
            same_sign_since = None
        if node.count_publishers('/cmd_vel') >= 2:                     # A8
            raise Abort('他の送り側が現れた')
        sub_count, method = node.sub_count()                           # A8b
        if sub_count != 1:
            raise Abort(f'接続異常: /cmd_vel の受け取り側が {sub_count} 個（{method}）'
                        '（この後の停止指令が届く保証はない）')
        if node.qos_incompatible:                                      # A9
            raise Abort('QoS 不一致の通知')
        if now - wall0 > SPIN_WALL_LIMIT:                              # A10
            raise Abort(f'SPIN が壁時計 {SPIN_WALL_LIMIT}s を超えた')
        if now - node.t_start > TOTAL_WALL_LIMIT:
            raise Abort(f'全体が壁時計 {TOTAL_WALL_LIMIT}s を超えた')
        if node.sim_now - sim0 >= SPIN_SIM_S:
            return


def stop_and_verify(node):
    """速度0を STOP_WALL_S 秒送り、静止を確認する。(停止を確認できたか, 理由) を返す。
    送信成功は相手に届いた保証ではない。停止の判断は wait_still の観測だけで行う。"""
    period = 1.0 / RATE_HZ
    with ignore_signals():
        end = time.monotonic() + STOP_WALL_S
        tick = time.monotonic()
        while time.monotonic() < end:                                  # 壁時計で必ず終わる
            node.send_zero()
            tick = advance_tick(tick, period)                          # [R2]
            node.safe_pump_until(min(tick, end))                       # [R1] 受信失敗でも送信は続ける
    try:
        node.report['sub_count_after_stop'] = node.sub_count()
    except Exception as e:   # noqa: BLE001
        node.report['sub_count_after_stop'] = f'取得失敗: {e!r}'
    node.report.update(zero_ok=node.zero_ok, zero_fail=node.zero_fail)
    if node.pump_error:
        node.report['pump_error'] = node.pump_error
    if node.zero_ok == 0:
        return False, f'速度0を1回も送れていない: {node.last_send_error}'
    ok, why, last = node.wait_still(VERIFY_SIM_TIMEOUT, VERIFY_WALL_TIMEOUT, send_zero_hz=VERIFY_ZERO_HZ)
    node.report['verify_last'] = last
    if node.zero_fail > 0:
        return False, f'速度0の送信失敗が {node.zero_fail} 回ある: {node.last_send_error}'
    return ok, why


def print_report(node, state, reason):
    print('=' * 60)
    print(f'結果: {state}')
    print(f'理由: {reason}')
    if node is not None:
        for k, v in node.report.items():
            print(f'  {k}: {v}')
        print(f'  速度0 送信成功/失敗: {node.zero_ok}/{node.zero_fail}')
        print('  注: 「送信成功」はミドルウェアに渡せたという意味で、相手に届いた保証ではない')
    if state == 'REPORT_FAIL':
        print('!' * 60)
        print('停止を確認できていない。Gazebo の画面で回っているか確認すること。')
        print('対応: (1) 速度0の手動送信（別途承認） (2) Gazebo GUI で Reset World（出現位置に戻る）')
        print('一時停止中なら、再開すると残った指令で再び回る可能性がある。')
        print('!' * 60)
    elif state == 'PRECHECK_NOT_STILL':
        print('!' * 60)
        print('このスクリプトは /cmd_vel に一度も送っていない。')
        print('ただし開始前にロボットが止まっていることを確認できていない（停止未確認）。')
        print('以前の指令が残っている、またはデータが届かない可能性がある。')
        print('Gazebo の画面で動いているか確認すること。')
        print('対応: (1) 速度0の手動送信（別途承認） (2) Gazebo GUI で Reset World（出現位置に戻る）')
        print('!' * 60)
    print('=' * 60)


MODES = ('--dry-run', '--spin')


def parse_mode(argv):
    """(mode, ROS 2 に渡す引数, エラー理由) を返す。

    '--ros-args' から '--' まで（'--' がなければ末尾まで）を ROS 2 の引数区間とし、
    '--' も含めてそのまま rclpy.init に渡す。'--' より後ろは、このスクリプトの引数に戻る。
    ROS 引数区間の外にある '--' は、このスクリプトの引数として数える（その場合は不正）。
    このスクリプトの引数は MODES のどちらか1つだけを許可し、それ以外はすべて不正とする。
    """
    user, ros = [], []
    in_ros = False
    for a in argv[1:]:
        if a == '--ros-args':
            in_ros = True
            ros.append(a)
        elif in_ros and a == '--':
            in_ros = False
            ros.append(a)
        elif in_ros:
            ros.append(a)
        else:
            user.append(a)
    if len(user) != 1 or user[0] not in MODES:
        return None, None, (f'起動引数が不正: {user}。'
                            f'--dry-run か --spin のどちらか1つだけを指定すること')
    return user[0], [argv[0]] + ros, None


def main():
    mode, ros_argv, arg_error = parse_mode(sys.argv)
    if arg_error is not None:
        print_report(None, 'ARG_ERROR', arg_error + '（ROS 2 初期化前に終了・送信なし）')
        sys.exit(EXIT_CODE['ARG_ERROR'])
    dry_run = (mode == '--dry-run')
    if SignalHandlerOptions is None:
        print_report(None, 'PRECHECK_ABORT',
                     'rclpy.signals.SignalHandlerOptions を利用できない。Ctrl+C 時に停止指令を送れるか'
                     '保証できないため、ノードを作らずに終了する（送信なし）。')
        sys.exit(EXIT_CODE['PRECHECK_ABORT'])
    args = ros_argv + ['--ros-args', '--disable-external-lib-logs']
    try:                                                               # [R4]
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)   # Ctrl+C を自前で扱う
        signal.signal(signal.SIGTERM, _on_sigterm)
        node = SpinAndStop()
    except BaseException as e:   # noqa: BLE001
        print_report(None, 'PRECHECK_ABORT', f'初期化に失敗（送信なし）: {type(e).__name__}: {e}')
        try:
            rclpy.shutdown()
        except Exception:   # noqa: BLE001
            pass
        sys.exit(EXIT_CODE['PRECHECK_ABORT'])

    state, reason = 'REPORT_FAIL', '未分類'
    try:
        ok, reason, fail_state = precheck(node, dry_run)
        if not ok:
            state = fail_state
        elif dry_run:
            state = 'DRYRUN_OK'
        else:
            aborted = None
            try:
                spin(node)
            except Abort as e:
                aborted = str(e)
            stopped, why = stop_and_verify(node)
            if not stopped:
                state = 'REPORT_FAIL'
                reason = why if aborted is None else f'{aborted} / {why}'
            elif aborted:
                state, reason = 'ABORT', aborted
            else:
                state, reason = 'DONE', 'OK'
    except BaseException as e:   # KeyboardInterrupt・Terminated・その他の例外
        reason = f'{type(e).__name__}: {e}'
        if not isinstance(e, (KeyboardInterrupt, Terminated)):
            traceback.print_exc()
        if not node.armed:
            state = 'PRECHECK_ABORT'   # まだ一度も送っていない
        else:
            try:
                stopped, why = stop_and_verify(node)   # 旋回開始後の異常は必ず STOP → VERIFY
                state = 'ABORT' if stopped else 'REPORT_FAIL'
                if not stopped:
                    reason += f' / {why}'
            except BaseException as e2:
                state, reason = 'REPORT_FAIL', reason + f' / 停止処理中に {type(e2).__name__}: {e2}'
    finally:
        with ignore_signals():                                         # [R3] finally 全体を保護
            if node.armed:
                # 念のための追加送信。停止の判定には使わない（判定は VERIFY のみ）。
                # 送信に失敗した場合だけ、成功扱いを取り消す。
                before_fail = node.zero_fail
                for _ in range(FINAL_ZERO_COUNT):
                    node.send_zero()
                    time.sleep(FINAL_ZERO_INTERVAL)
                if node.zero_fail > before_fail and state in ('DONE', 'ABORT'):
                    state, reason = 'REPORT_FAIL', reason + ' / 最後の速度0送信に失敗'
            print_report(node, state, reason)
            try:
                node.destroy_node()
            except Exception:   # noqa: BLE001
                pass
            try:
                rclpy.shutdown()
            except Exception:   # noqa: BLE001
                pass
    sys.exit(EXIT_CODE[state])


if __name__ == '__main__':
    main()
