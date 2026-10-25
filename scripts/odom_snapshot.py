#!/usr/bin/env python3
"""odom_snapshot.py : /odom からロボットの位置・向きを1回だけ取得して表示する（読み取り専用）。

しないこと：/cmd_vel の購読・送信、publisher の作成、ロボットの移動、Gazebo の操作、ファイルの書き込み。
rclpy は ros_collect() の中でだけ import する（ホストで ROS なしにテストできるようにするため）。

判定（すべて暫定値）
  - 有効なサンプル（/odom）が2件以上。header.stamp が前より増えたものだけを別のサンプルとして数える
    （同じ時刻は重複として数えない。後戻りは即座に失敗）。
  - 位置・四元数・時刻はすべて有限。四元数の大きさは 1 ± QUAT_NORM_TOL。
  - /clock を2回以上受信し、値が増えていること（進んでいなければ鮮度を確認できないので成功にしない。
    同じ値は進んでいない扱い、後戻りは即座に失敗。spin_and_stop.py も /clock は増えたときだけ更新している）。
  - 鮮度 age = 最新の /clock − 最新の /odom の header.stamp が 0 以上 MAX_AGE_S 以下。
    age が負（/odom が /clock より先に届いた）や MAX_AGE_S 超は、その場で確定せず新しいサンプルを待ち、
    受信待ちの上限までに整合しなければ失敗。
  - 失敗のときは ODOM_SNAPSHOT の行を出さない（yaw=0 などと取り違えないため）。

出力（成功時）: 'ODOM_SNAPSHOT {json}' の1行と '結果: OK'
終了コード: 0=OK  2=引数の誤り  6=INIT_ERROR  7=NO_MESSAGE（受信なし・サンプル不足）  8=INVALID
            9=STALE / FUTURE_STAMP / TIME_INCONSISTENT / NO_CLOCK / CLOCK_NOT_ADVANCING  11=INTERRUPTED
将来 mission_runner.py から使うときは make_heading_provider() を heading に渡す（mission_runner.py は変更しない）。
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass

QUAT_NORM_TOL = 1e-3      # 四元数の大きさの許容（暫定）
MAX_AGE_S = 0.5           # s(sim) 鮮度の上限（暫定。spin_and_stop.py の STALE_WALL と同じ値）
TIMEOUT_S = 5.0           # s(wall) 受信待ちの上限（暫定。spin_and_stop.py の PRECHECK_WALL_TIMEOUT と同じ値）
MIN_SAMPLES = 2
SPIN_STEP_S = 0.05
LINE_TAG = 'ODOM_SNAPSHOT '

EXIT = {'OK': 0, 'ARG_ERROR': 2, 'INIT_ERROR': 6, 'NO_MESSAGE': 7, 'INVALID': 8, 'STALE': 9, 'FUTURE_STAMP': 9,
        'TIME_INCONSISTENT': 9, 'NO_CLOCK': 9, 'CLOCK_NOT_ADVANCING': 9, 'INTERRUPTED': 11}


class SnapshotError(ValueError):
    """値の不正（四元数など）"""


class SnapshotUnavailable(Exception):
    """make_heading_provider() で向きを得られない。mission_runner はどんな例外でも END_UNDETERMINED にする。"""


def _finite(*vals):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def yaw_from_quat(x, y, z, w):
    """spin_and_stop.py / person_geometry.py と同じ式。範囲は [-π, π]。不正な値は SnapshotError。"""
    if not _finite(x, y, z, w):
        raise SnapshotError('四元数に有限でない値がある')
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if abs(n - 1.0) > QUAT_NORM_TOL:
        raise SnapshotError(f'四元数の大きさが 1 ± {QUAT_NORM_TOL} の外: {n:.6f}')
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass
class OdomSample:
    stamp: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    frame_id: str = ''
    child_frame_id: str = ''
    wall: float = 0.0          # 受信した壁時計（time.monotonic）


@dataclass
class Snapshot:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    yaw: float
    stamp: float
    clock: float
    age: float
    n_samples: int
    n_duplicates: int
    frame_id: str
    child_frame_id: str
    wall_since_receive: float


class SnapshotCollector:
    """受信したサンプルを検査して、状態を返す。ROS に依存しない。
    status() は ('OK', Snapshot, 理由) / ('PENDING', 状態名, 理由) / ('FAIL', 状態名, 理由) のどれか。"""

    def __init__(self, now=time.monotonic):
        self.now = now
        self.samples = []
        self.duplicates = 0
        self.clocks = []           # 増えた値だけ
        self.hard = None           # (状態名, 理由)：即座に失敗が確定

    def _fail(self, state, why):
        if self.hard is None:
            self.hard = (state, why)

    def add_odom(self, s):
        if self.hard:
            return
        if not _finite(s.stamp, s.x, s.y, s.z):
            return self._fail('INVALID', '位置または時刻に有限でない値がある')
        try:
            yaw_from_quat(s.qx, s.qy, s.qz, s.qw)
        except SnapshotError as e:
            return self._fail('INVALID', str(e))
        if self.samples:
            last = self.samples[-1].stamp
            if s.stamp < last:
                return self._fail('TIME_INCONSISTENT', f'/odom の時刻が後戻り: {last} → {s.stamp}')
            if s.stamp == last:
                self.duplicates += 1
                return
        self.samples.append(s)

    def add_clock(self, t):
        if self.hard:
            return
        if not _finite(t):
            return self._fail('INVALID', '/clock に有限でない値がある')
        if self.clocks:
            if t < self.clocks[-1]:
                return self._fail('TIME_INCONSISTENT', f'/clock が後戻り: {self.clocks[-1]} → {t}')
            if t == self.clocks[-1]:
                return
        self.clocks.append(t)

    def status(self):
        if self.hard:
            return ('FAIL',) + self.hard
        if not self.samples:
            return 'PENDING', 'NO_MESSAGE', '/odom を受信していない'
        if len(self.samples) < MIN_SAMPLES:
            return 'PENDING', 'NO_MESSAGE', f'時刻の異なる /odom が {len(self.samples)} 件（{MIN_SAMPLES} 件以上必要）'
        if not self.clocks:
            return 'PENDING', 'NO_CLOCK', '/clock を受信していない'
        if len(self.clocks) < 2:
            return 'PENDING', 'CLOCK_NOT_ADVANCING', '/clock が進んでいることを確認できていない'
        s, clock = self.samples[-1], self.clocks[-1]
        age = clock - s.stamp
        if age < 0:
            return 'PENDING', 'FUTURE_STAMP', f'/odom の時刻が /clock より {-age:.3f}s 先（新しい /clock を待つ）'
        if age > MAX_AGE_S:
            return 'PENDING', 'STALE', f'/odom が {age:.3f}s 古い（上限 {MAX_AGE_S}s）'
        snap = Snapshot(s.x, s.y, s.z, s.qx, s.qy, s.qz, s.qw, yaw_from_quat(s.qx, s.qy, s.qz, s.qw), s.stamp, clock,
                        age, len(self.samples), self.duplicates, s.frame_id, s.child_frame_id, self.now() - s.wall)
        return 'OK', snap, 'OK'


def run_collection(collector, spin_once, timeout=TIMEOUT_S, now=time.monotonic):
    """spin_once(秒) で受信を進め、OK か即座の失敗が出るか、時間切れまで待つ。
    戻り値 ('OK', Snapshot, 理由) または ('FAIL', 状態名, 理由)。時間切れは待っていた状態名で失敗にする。"""
    start = now()
    while True:
        spin_once(SPIN_STEP_S)
        st = collector.status()
        if st[0] in ('OK', 'FAIL'):
            return st
        if now() - start >= timeout:
            return 'FAIL', st[1], f'{timeout}s 以内に条件を満たさない: {st[2]}'


def ros_collect(timeout=TIMEOUT_S):
    """ROS 2 で /odom と /clock を購読して1回分を取得する。publisher は作らない。"""
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock

    collector = SnapshotCollector()

    def on_odom(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        collector.add_odom(OdomSample(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, p.x, p.y, p.z,
                                      q.x, q.y, q.z, q.w, msg.header.frame_id, msg.child_frame_id, time.monotonic()))

    def on_clock(msg):
        collector.add_clock(msg.clock.sec + msg.clock.nanosec * 1e-9)

    rclpy.init(args=['odom_snapshot', '--ros-args', '--disable-external-lib-logs'])
    node = None
    try:
        node = rclpy.create_node('odom_snapshot')
        node.create_subscription(Odometry, '/odom', on_odom, qos_profile_sensor_data)
        node.create_subscription(Clock, '/clock', on_clock, qos_profile_sensor_data)
        return run_collection(collector, lambda t: rclpy.spin_once(node, timeout_sec=t), timeout)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def format_line(snap):
    return LINE_TAG + json.dumps(asdict(snap), ensure_ascii=False, sort_keys=True)


def main(argv=None, collect=ros_collect, out=sys.stdout):
    p = argparse.ArgumentParser(description='/odom の位置・向きを1回だけ取得する（読み取り専用）')
    p.add_argument('--timeout', type=float, default=TIMEOUT_S, help='受信待ちの上限 [s(wall)]')
    args = p.parse_args(argv)
    if not (math.isfinite(args.timeout) and 0.0 < args.timeout <= 60.0):
        p.error('--timeout は 0 より大きく 60 以下にすること')
    try:
        st = collect(args.timeout)
    except KeyboardInterrupt:
        st = ('FAIL', 'INTERRUPTED', 'KeyboardInterrupt')
    except ImportError as e:
        st = ('FAIL', 'INIT_ERROR', f'import に失敗: {e}')
    except Exception as e:  # noqa: BLE001
        st = ('FAIL', 'INIT_ERROR', f'{type(e).__name__}: {e}')
    if st[0] == 'OK':
        snap = st[1]
        print(f'位置 ({snap.x:.4f}, {snap.y:.4f})  yaw {snap.yaw:.4f} rad ({math.degrees(snap.yaw):.2f}°)  '
              f'鮮度 {snap.age:.3f}s  サンプル {snap.n_samples} 件', file=out)
        print(format_line(snap), file=out)
        print('結果: OK', file=out)
        return EXIT['OK']
    print(f'結果: {st[1]}', file=out)
    print(f'詳細: {st[2]}', file=out)
    return EXIT[st[1]]


# ---------------- mission_runner.py との接続 ----------------
def parse_snapshot_output(output):
    """子プロセスの出力から Snapshot の dict を取り出す。条件を満たさなければ SnapshotUnavailable。"""
    results = [l[len('結果: '):].strip() for l in output.splitlines() if l.startswith('結果: ')]
    if not results or results[-1] != 'OK':
        raise SnapshotUnavailable(f'結果が OK ではない: {results[-1] if results else "なし"}')
    lines = [l for l in output.splitlines() if l.startswith(LINE_TAG)]
    if len(lines) != 1:
        raise SnapshotUnavailable(f'{LINE_TAG.strip()} の行が {len(lines)} 個（1 個のはず）')
    try:
        d = json.loads(lines[0][len(LINE_TAG):])
    except ValueError as e:
        raise SnapshotUnavailable(f'JSON を読めない: {e}') from None
    need = ('x', 'y', 'qx', 'qy', 'qz', 'qw', 'yaw', 'stamp', 'clock', 'age', 'n_samples')
    if not isinstance(d, dict) or any(k not in d for k in need):
        raise SnapshotUnavailable('必要な項目が足りない')
    if not _finite(*(d[k] for k in need if k != 'n_samples')):
        raise SnapshotUnavailable('有限でない値がある')
    try:
        yaw = yaw_from_quat(d['qx'], d['qy'], d['qz'], d['qw'])
    except SnapshotError as e:
        raise SnapshotUnavailable(str(e)) from None
    if abs(yaw - d['yaw']) > 1e-6 or abs(d['yaw']) > math.pi:
        raise SnapshotUnavailable('yaw が四元数と合わない、または範囲外')
    if not (0.0 <= d['age'] <= MAX_AGE_S) or not (isinstance(d['n_samples'], int) and d['n_samples'] >= MIN_SAMPLES):
        raise SnapshotUnavailable('鮮度またはサンプル数が条件を満たさない')
    return d


def make_heading_provider(run_child, python='python3', script='/home/ubuntu/odom_snapshot.py'):
    """mission_runner.MissionRunner(heading=...) に渡す関数を返す。
    run_child(cmd) は mission_runner.ChildResult と同じ属性（returncode, output, started）を持つ値を返すこと。
    子の終了コード 0・'結果: OK'・正しい ODOM_SNAPSHOT 行がそろったときだけ yaw [rad] を返し、それ以外は例外。"""
    def heading(round_no):
        r = run_child([python, '-B', script])
        if not getattr(r, 'started', False) or r.returncode != 0:
            raise SnapshotUnavailable(f'odom_snapshot.py の終了コード {r.returncode}（起動 {getattr(r, "started", None)}）')
        return parse_snapshot_output(r.output)['yaw']
    return heading


if __name__ == '__main__':
    sys.exit(main())
