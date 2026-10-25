#!/usr/bin/env python3
"""mission_logic.py : 探索結果（search_loop.py の CSV）を統合し、次の行動を判定する純粋な関数（STEP 19-1）。

ROS 2・Gazebo・YOLO には依存しない。/cmd_vel は扱わない。ファイルは読むだけ（load_search_csv）。
既存の search_loop.py / person_geometry.py / spin_and_stop.py は変更せず、列名だけ search_loop から読む。

用語（主張の範囲）
  観測ラウンド : search_loop.py の 1 回の実行（1 つの CSV）。
  DETECTED    : そのクラスタに LOCATED の観測が 1 回以上ある（報告の根拠にならない観測だけの場合も含む）。
  CANDIDATE   : 根拠になる観測が MIN_EVIDENCE 回以上あるが、REPORTED の条件を満たしていない。
  REPORTED    : シミュレーション上で、ここで定義した確認条件を満たしたという意味だけを持つ。
                実際の要救助者の発見や位置の正しさを保証するものではない。
  NO_PERSON   : その周で人物が検出されなかったという記録。その向きに人物がいないことは意味しない。
  一周         : 明示的に与えられた各ラウンドの向き（/odom の ψ）と順序から求めた累積の回転角で判定する。
                CSV の件数やファイル名からは推定しない。扇形はカメラの幾何学的な画角で、検出できる範囲ではない。

次の行動（decision）
  DONE_REPORTED   : REPORTED のクラスタがある。
  RESEARCH        : 旋回せず、同じ向きでもう一度探索する（候補の再観測、または異常のあった観測のやり直し）。
  SPIN_NEXT       : 次の向きへ旋回する（spin_and_stop.py の固定量の左回り）。
  DONE_NOT_FOUND  : この場所から一周観測したが、確認条件を満たすクラスタがない（不在の証明ではない）。
  DONE_SPIN_LIMIT : 旋回の上限に達したが、一周の条件を満たしていない。
  UNDETERMINED    : 順序・向き・旋回の有無の情報が不足しており、次の行動を判定できない。
  ABORT           : 入力の不正・不整合。報告も終了判定もしない。
"""
import csv
import math
import statistics
from dataclasses import dataclass, field

from search_loop import RAW_FIELDS

# ---------------- 暫定値（すべて STEP 15〜17 のシミュレーションだけを根拠にした仮の値） ----------------
MIN_CONF = 0.5                 # 報告の根拠にする信頼度の下限
MERGE_RADIUS = 0.5             # m  クラスタ平均からこの距離以内ならまとめる（境界の値はまとめる）
MAX_STD = 0.15                 # m  REPORTED に必要な座標ごとの標本標準偏差の上限（境界の値は通す）
SPIN_LIMIT = 7                 # 旋回の上限回数
MIN_EVIDENCE = 3               # CANDIDATE・REPORTED に必要な根拠の観測数
MIN_EVIDENCE_ROUNDS = 2        # REPORTED に必要な、根拠の観測を含む別々の観測ラウンドの数
MAX_ROUNDS_PER_HEADING = 3     # 同じ向きで探索する回数の上限（初回を含む）
MIN_INFERENCE_CYCLES = 2       # その向きを「観測済み」とするのに必要な推論済みの周の数
VERIFIED_Z_RANGE = (2.6, 3.0)  # m  検証済みの距離A。STEP 15〜17 の LOCATED の z_a 2.68〜2.81m と真値 2.82〜2.92m を含む範囲
HFOV_RAD = 1.085595            # rad 水平画角（/camera/camera_info の fx=530.467・幅640 からの計算値）
MAX_SPIN_STEP_RAD = 1.05       # rad 1 回の旋回の上限（spin_and_stop.py の ABORT_OVERROT と同じ値。import はしない）
HEADING_TOL_RAD = 0.02         # rad 旋回なしのラウンド間の向きの差、CSV の psi_deg と与えた向きの差の許容
POS_TOL = 0.05                 # m  その場旋回の前提。最初の位置からのずれの許容（spin_and_stop.py の ABORT_POS と同じ値）
FULL_TURN_RAD = 2.0 * math.pi
EPS = 1e-9                     # 浮動小数点の比較の余裕

EDGE_FLAGS = {'SIDE_CUT', 'TOP_CUT', 'BOTTOM_CUT', 'ABOVE_HORIZON'}   # person_geometry.edge_flags の名前
INFERENCE_STATUSES = {'LOCATED', 'BEARING_ONLY', 'NO_PERSON'}       # 推論まで行った周
ABNORMAL_STATUSES = {'NO_IMAGE', 'NO_ODOM', 'ODOM_TIME_MISMATCH', 'CYCLE_ERROR'}  # 正常な未検出ではない
KNOWN_STATUSES = INFERENCE_STATUSES | ABNORMAL_STATUSES
REQUIRED_NUMERIC = {
    'LOCATED': ('conf', 'u_c', 'phi_deg', 'z_a', 'xr', 'yr', 'psi_deg', 'X', 'Y'),
    'BEARING_ONLY': ('conf', 'u_c', 'phi_deg', 'xr', 'yr', 'psi_deg'),
    'NO_PERSON': ('xr', 'yr', 'psi_deg'),
}


class InputError(ValueError):
    """入力 CSV・ラウンド情報の不正"""


def wrap(a):
    """角度を [-π, π) に折り返す。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class RoundInput:
    """1 回の観測ラウンド。order・psi_rad・after_spin は呼び出し側が明示的に与える（CSV からは推定しない）。"""
    round_id: str
    rows: list
    order: int = None          # 1 から始まる観測の順番
    psi_rad: float = None      # このラウンドのときのロボットの向き（/odom、旋回後の値）
    after_spin: bool = None    # 直前に旋回したか（False = 同じ向きでの再観測）
    source: str = None         # 読み込んだファイルのパス（重複の検出用）


@dataclass
class Observation:
    round_id: str
    cycle: int
    status: str
    img_stamp: str
    conf: float = None
    X: float = None
    Y: float = None
    z_a: float = None
    psi_rad: float = None
    xr: float = None
    yr: float = None
    flags: tuple = ()
    excluded: list = field(default_factory=list)   # 報告の根拠から外した理由（記録は残す）
    unverified_distance: bool = False
    cluster_id: int = None

    @property
    def is_evidence(self):
        return self.status == 'LOCATED' and not self.excluded


@dataclass
class Cluster:
    cluster_id: int
    evidence: list = field(default_factory=list)   # 平均・標準偏差・状態の計算に使う観測
    records: list = field(default_factory=list)    # 根拠から外した LOCATED（記録だけ）
    state: str = 'DETECTED'
    notes: list = field(default_factory=list)

    def mean(self):
        return (statistics.fmean(o.X for o in self.evidence), statistics.fmean(o.Y for o in self.evidence))

    def std(self):
        if len(self.evidence) < 2:
            return None
        return (statistics.stdev(o.X for o in self.evidence), statistics.stdev(o.Y for o in self.evidence))

    def evidence_rounds(self):
        return sorted({o.round_id for o in self.evidence})


@dataclass
class RoundSummary:
    round_id: str
    order: int
    psi_rad: float
    after_spin: bool
    counts: dict
    observed: bool             # その向きを観測済みとして一周の判定に数えるか
    unwrapped_rad: float = None


@dataclass
class Evaluation:
    decision: str
    reasons: list
    clusters: list = field(default_factory=list)
    rounds: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    cumulative_rad: float = None
    full_turn: bool = False
    spins: int = 0
    notes: list = field(default_factory=lambda: [
        'NO_PERSON はその向きに人物がいないことを意味しない',
        'REPORTED はシミュレーション上で定義した確認条件を満たしたことだけを意味し、実際の発見を保証しない',
        'クラスタは位置の近さでまとめたもので、同じ人物であることは保証しない',
        '扇形の画角は幾何学的な値で、検出できる範囲ではない（検出の境界は未測定）',
    ])


# ---------------- 入力 ----------------
def load_search_csv(path):
    """search_loop.py の 1 周ごとの CSV を読む。列不足・空ファイルは InputError。行の中身は parse_row で検査する。"""
    with open(path, newline='', encoding='utf-8') as f:
        return read_search_csv(f, path)


def read_search_csv(f, name):
    """開いたファイル（または io.StringIO）から読む。name はエラー表示用。"""
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
        raise InputError(f'{name}: ヘッダーがない（空ファイル）')
    missing = [c for c in RAW_FIELDS if c not in reader.fieldnames]
    if missing:
        raise InputError(f'{name}: 列が足りない: {missing}')
    rows = []
    for i, row in enumerate(reader, start=2):
        if None in row or any(v is None for v in row.values()):   # 列が多すぎる行・足りない行
            raise InputError(f'{name}: {i} 行目の列数がヘッダーと合わない')
        rows.append(row)
    return rows


def _num(row, key, where):
    v = row.get(key)
    if v is None or v == '':
        raise InputError(f'{where}: {key} が空')
    try:
        x = float(v)
    except ValueError:
        raise InputError(f'{where}: {key} が数値ではない: {v!r}') from None
    if not math.isfinite(x):
        raise InputError(f'{where}: {key} が有限の数ではない: {v!r}')
    return x


def parse_row(row, round_id):
    """1 行を Observation にする。不正なら InputError。"""
    missing = [c for c in RAW_FIELDS if c not in row]
    if missing:
        raise InputError(f'{round_id}: 列が足りない: {missing}')
    try:
        cycle = int(row['cycle'])
    except (TypeError, ValueError):
        raise InputError(f'{round_id}: cycle が整数ではない: {row["cycle"]!r}') from None
    if cycle < 1:
        raise InputError(f'{round_id}: cycle が 1 未満: {cycle}')
    where = f'{round_id} 周{cycle}'
    status = row['status']
    if status not in KNOWN_STATUSES:
        raise InputError(f'{where}: 不明な status: {status!r}')
    ob = Observation(round_id=round_id, cycle=cycle, status=status, img_stamp=row['img_stamp'] or '')
    vals = {k: _num(row, k, where) for k in REQUIRED_NUMERIC.get(status, ())}
    if 'psi_deg' in vals:
        ob.psi_rad = math.radians(vals['psi_deg'])
        ob.xr, ob.yr = vals['xr'], vals['yr']
    if status == 'NO_PERSON':
        if row['n_det'] not in ('0', ''):
            raise InputError(f'{where}: NO_PERSON なのに n_det={row["n_det"]!r}')
        return ob
    if status in ('LOCATED', 'BEARING_ONLY'):
        ob.conf = vals['conf']
        if not 0.0 <= ob.conf <= 1.0:
            raise InputError(f'{where}: conf が 0〜1 の外: {ob.conf}')
        ob.flags = tuple(f for f in (row['flags'] or '').split('|') if f)
    if status == 'LOCATED':
        ob.X, ob.Y, ob.z_a = vals['X'], vals['Y'], vals['z_a']
        if ob.z_a <= 0.0:
            raise InputError(f'{where}: z_a が正ではない: {ob.z_a}')
        for f in ob.flags:
            ob.excluded.append(f'印 {f}' if f in EDGE_FLAGS else f'不明な印 {f}')
        if ob.conf < MIN_CONF:
            ob.excluded.append(f'信頼度 {ob.conf:.3f} < {MIN_CONF}')
        lo, hi = VERIFIED_Z_RANGE
        ob.unverified_distance = not (lo <= ob.z_a <= hi)
    return ob


# ---------------- クラスタ ----------------
def build_clusters(observations):
    """根拠になる観測を、与えられた順に平均からの距離でまとめる（search_loop.PositionClusters と同じ方式。順序に依存する）。
    根拠から外した LOCATED は、最も近いクラスタ（半径以内）に記録だけ付ける。どこにも入らなければ記録だけのクラスタを作る。"""
    clusters = []

    def nearest(o, pool):
        best, best_d = None, None
        for c in pool:
            mx, my = c.mean()
            d = math.hypot(o.X - mx, o.Y - my)
            if d <= MERGE_RADIUS + EPS and (best_d is None or d < best_d):
                best, best_d = c, d
        return best

    for o in observations:
        if not o.is_evidence:
            continue
        c = nearest(o, clusters)
        if c is None:
            c = Cluster(cluster_id=len(clusters) + 1)
            clusters.append(c)
        c.evidence.append(o)
        o.cluster_id = c.cluster_id
    for o in observations:
        if o.status != 'LOCATED' or o.is_evidence:
            continue
        c = nearest(o, [c for c in clusters if c.evidence])
        if c is None:
            c = next((c for c in clusters if not c.evidence and c.records
                      and math.hypot(o.X - c.records[0].X, o.Y - c.records[0].Y) <= MERGE_RADIUS + EPS), None)
        if c is None:
            c = Cluster(cluster_id=len(clusters) + 1)
            clusters.append(c)
        c.records.append(o)
        o.cluster_id = c.cluster_id
    for c in clusters:
        classify_cluster(c)
    return clusters


def classify_cluster(c):
    n = len(c.evidence)
    rounds = c.evidence_rounds()
    std = c.std()
    if any(o.unverified_distance for o in c.evidence + c.records):
        c.notes.append(f'未検証距離の観測を含む（検証済みは距離A {VERIFIED_Z_RANGE[0]}〜{VERIFIED_Z_RANGE[1]} m だけ）')
    if c.records:
        c.notes.append(f'根拠から外した観測 {len(c.records)} 件（記録のみ）')
    if n < MIN_EVIDENCE:
        c.state = 'DETECTED'
        c.notes.append(f'根拠の観測 {n} 件 < {MIN_EVIDENCE}')
        return
    c.state = 'CANDIDATE'
    if len(rounds) < MIN_EVIDENCE_ROUNDS:
        c.notes.append(f'別の観測ラウンドでの再観測がない（ラウンド {rounds}）')
        return
    if std is None or std[0] > MAX_STD + EPS or std[1] > MAX_STD + EPS:
        c.notes.append(f'標準偏差 {std} が {MAX_STD} m を超える')
        return
    c.state = 'REPORTED'


# ---------------- 判定 ----------------
def _abort(errors, rounds=None, clusters=None):
    return Evaluation(decision='ABORT', reasons=['入力の不正・不整合のため、報告も終了判定もしない'],
                      errors=errors, rounds=rounds or [], clusters=clusters or [])


def evaluate(rounds):
    """RoundInput のリストから、クラスタの状態と次の行動を判定する。例外は投げず、不正は ABORT として返す。"""
    errors = []
    if not rounds:
        return _abort(['観測ラウンドがない（空データ）'])
    ids = [r.round_id for r in rounds]
    if len(set(ids)) != len(ids):
        errors.append(f'round_id の重複: {ids}')
    srcs = [r.source for r in rounds if r.source is not None]
    if len(set(srcs)) != len(srcs):
        errors.append(f'同じファイルが複数回与えられている: {srcs}')

    per_round = {}
    for r in rounds:
        obs = []
        for row in r.rows:
            try:
                obs.append(parse_row(row, r.round_id))
            except InputError as e:
                errors.append(str(e))
        cycles = [o.cycle for o in obs]
        if len(set(cycles)) != len(cycles):
            errors.append(f'{r.round_id}: cycle の重複')
        per_round[r.round_id] = obs
    if errors:
        return _abort(errors)

    all_obs = [o for r in rounds for o in per_round[r.round_id]]
    stamps = [o.img_stamp for o in all_obs if o.img_stamp]
    dup = sorted({s for s in stamps if stamps.count(s) > 1})
    if dup:
        errors.append(f'同じ画像時刻の行が複数ある（重複入力の可能性）: {dup[:5]}')

    # 与えられた向きと CSV の psi_deg の一致、その場旋回の前提（位置）を確かめる
    pos = [(o.xr, o.yr) for o in all_obs if o.xr is not None]
    for o in all_obs:
        if o.xr is not None and math.hypot(o.xr - pos[0][0], o.yr - pos[0][1]) > POS_TOL + EPS:
            errors.append(f'{o.round_id} 周{o.cycle}: 位置が最初の観測から {POS_TOL} m を超えて動いている')
    for r in rounds:
        if r.psi_rad is None:
            continue
        if not isinstance(r.psi_rad, (int, float)) or not math.isfinite(r.psi_rad):
            errors.append(f'{r.round_id}: psi_rad が有限の数ではない')
            continue
        for o in per_round[r.round_id]:
            if o.psi_rad is not None and abs(wrap(o.psi_rad - r.psi_rad)) > HEADING_TOL_RAD + EPS:
                errors.append(f'{r.round_id} 周{o.cycle}: CSV の向きと与えた向きが {HEADING_TOL_RAD} rad を超えて違う')

    summaries = []
    for r in rounds:
        counts = {}
        for o in per_round[r.round_id]:
            counts[o.status] = counts.get(o.status, 0) + 1
        n_inf = sum(counts.get(s, 0) for s in INFERENCE_STATUSES)
        n_abn = sum(counts.get(s, 0) for s in ABNORMAL_STATUSES)
        summaries.append(RoundSummary(r.round_id, r.order, r.psi_rad, r.after_spin, counts,
                                      observed=(n_abn == 0 and n_inf >= MIN_INFERENCE_CYCLES)))

    ordered = all(r.order is not None and r.psi_rad is not None and r.after_spin is not None for r in rounds)
    if ordered:
        orders = sorted(r.order for r in rounds)
        if orders != list(range(1, len(rounds) + 1)):
            errors.append(f'order が 1 からの連番ではない: {orders}')
    if errors:
        return _abort(errors, summaries)

    clusters = build_clusters(all_obs)   # 与えられたリストの順（order があれば下で並べ替えてから再計算）
    if ordered:
        by_order = sorted(rounds, key=lambda r: r.order)
        all_obs = [o for r in by_order for o in per_round[r.round_id]]
        for o in all_obs:
            o.cluster_id = None
        clusters = build_clusters(all_obs)
        summaries.sort(key=lambda s: s.order)
        acc = 0.0
        for i, (r, s) in enumerate(zip(by_order, summaries)):
            if i > 0:
                d = wrap(r.psi_rad - by_order[i - 1].psi_rad)
                if r.after_spin:
                    if not (EPS < d <= MAX_SPIN_STEP_RAD + EPS):
                        errors.append(f'{r.round_id}: 旋回後の向きの変化 {d:.4f} rad が左回り 0〜{MAX_SPIN_STEP_RAD} rad の範囲外')
                    acc += d
                elif abs(d) > HEADING_TOL_RAD + EPS:
                    errors.append(f'{r.round_id}: 旋回なしのはずが向きが {d:.4f} rad 変わっている')
            s.unwrapped_rad = acc
        if errors:
            return _abort(errors, summaries, clusters)

    ev = Evaluation(decision='', reasons=[], clusters=clusters, rounds=summaries)
    reported = [c for c in clusters if c.state == 'REPORTED']
    if reported:
        ev.decision = 'DONE_REPORTED'
        ev.reasons.append(f'確認条件を満たしたクラスタ: {[c.cluster_id for c in reported]}（シミュレーション上の判定）')
        if ordered:
            _fill_coverage(ev, by_order, summaries)
        return ev
    if not ordered:
        ev.decision = 'UNDETERMINED'
        ev.reasons.append('order・psi_rad・after_spin のいずれかが与えられていないラウンドがあり、次の行動と一周を判定できない')
        return ev

    _fill_coverage(ev, by_order, summaries)
    # 最後の旋回以降のラウンド（いまの向き）
    last_spin = max([i for i, r in enumerate(by_order) if r.after_spin] or [0])
    here = by_order[last_spin:]
    here_ids = {r.round_id for r in here}
    can_research = len(here) < MAX_ROUNDS_PER_HEADING
    if not summaries[-1].observed:
        if can_research:
            ev.decision = 'RESEARCH'
            ev.reasons.append(f'{summaries[-1].round_id}: 異常のある周がある、または推論済みの周が {MIN_INFERENCE_CYCLES} 未満（正常な未検出ではない）。同じ向きでやり直す')
        else:
            ev.decision = 'ABORT'
            ev.reasons.append(f'同じ向きで {MAX_ROUNDS_PER_HEADING} 回探索しても正常な観測にならない')
        return ev
    here_evidence = [c.cluster_id for c in clusters if any(o.round_id in here_ids for o in c.evidence)]
    if here_evidence and can_research:
        ev.decision = 'RESEARCH'
        ev.reasons.append(f'いまの向きで根拠になる検出がある（クラスタ {here_evidence}）。別ラウンドで再観測する')
        return ev
    if here_evidence:
        ev.reasons.append(f'いまの向きで {MAX_ROUNDS_PER_HEADING} 回探索したが確認条件を満たさない（クラスタ {here_evidence}）')
    if ev.full_turn:
        ev.decision = 'DONE_NOT_FOUND'
        ev.reasons.append('この場所から一周観測したが、確認条件を満たすクラスタはない（不在の証明ではない）')
        return ev
    if ev.spins >= SPIN_LIMIT:
        ev.decision = 'DONE_SPIN_LIMIT'
        ev.reasons.append(f'旋回の上限 {SPIN_LIMIT} 回に達したが、一周の条件を満たしていない')
        return ev
    ev.decision = 'SPIN_NEXT'
    ev.reasons.append('いまの向きでは根拠になる検出がない。次の向きへ旋回する')
    return ev


def _fill_coverage(ev, by_order, summaries):
    """観測済みの向きの累積角で一周を判定する。観測済みの向きどうしの間隔が画角を超えたら一周とみなさない。"""
    ev.spins = sum(1 for r in by_order if r.after_spin)
    seen = [s.unwrapped_rad for s in summaries if s.observed]
    ev.cumulative_rad = summaries[-1].unwrapped_rad
    if len(seen) < 2:
        ev.full_turn = False
        return
    gaps = [b - a for a, b in zip(seen, seen[1:])]
    span = seen[-1] - seen[0]
    ev.full_turn = span + EPS >= FULL_TURN_RAD and max(gaps) <= HFOV_RAD + EPS
    if span + EPS >= FULL_TURN_RAD and not ev.full_turn:
        ev.reasons.append(f'累積 {span:.3f} rad だが、観測済みの向きの間隔が画角 {HFOV_RAD} rad を超える箇所がある')
