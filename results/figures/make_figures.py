#!/usr/bin/env python3
"""make_figures.py : STEP 15〜17 の CSV（results/step15-17/）から F1・F2・F3 の SVG/HTML と計算表を作る。

入力の CSV は読み取るだけで、変更しない。標準ライブラリだけを使う。
出力（このフォルダ）: f1_headings.svg, f2_positions.svg, f3_errors.svg, figures.html, metrics_step15-17.csv
PNG は別途、ブラウザで SVG を画像化して作る（README 参照）。

真値・定数の出典（CSV の外にある値）
  人物の真値 (1.0, -0.9)      : house_with_person_v4.world のアクター基準点（設定値）
  カメラの前方オフセット 0.08 m : happy_mini_camfix.sdf のカメラリンク位置（STEP 13 で gz model -i により 0.080 を確認）
  fx = 530.467 / 画像幅 640   : /camera/camera_info（STEP 14・15 の実行時出力。CSV には含まれない）
"""
import csv
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, '..', 'step15-17'))

GT = (1.0, -0.9)
CAM_X = 0.08
FX = 530.467
IMG_W = 640
HALF_FOV = math.degrees(math.atan((IMG_W / 2) / FX))  # 幾何学的な水平半画角（計算値）

RUNS = [  # (CSV の時刻, 表示名, 内容)
    ('164652', 'T2', 'STEP 15 T2（向き 0°）'),
    ('170016', 'S1', 'STEP 16 S1（向き 0°）'),
    ('171320', 'S4', 'STEP 16 S4'),
    ('173859', 'k=1', 'STEP 17 k=1'),
    ('174245', 'k=2', 'STEP 17 k=2'),
    ('174910', 'k=3', 'STEP 17 k=3'),
    ('175222', 'k=4', 'STEP 17 k=4'),
    ('175508', 'k=5', 'STEP 17 k=5'),
    ('175735', 'k=6', 'STEP 17 k=6（向き 約8.9°）'),
]
LOC_RUNS = ['T2', 'S1', 'k=6']
SERIES = {'T2': ('#2a78d6', 'circle'), 'S1': ('#eb6834', 'square'), 'k=6': ('#1baf7a', 'triangle')}

INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
FONT = "'Yu Gothic UI','Meiryo','Segoe UI',system-ui,sans-serif"


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def read_csv(name):
    with open(os.path.join(SRC, name), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ---------- 読み込み ----------
runs = []
for tag, label, desc in RUNS:
    rows = read_csv(f'search_20261024_{tag}.csv')
    cl = read_csv(f'search_20261024_{tag}_clusters.csv')
    runs.append({'tag': tag, 'label': label, 'desc': desc, 'rows': rows, 'clusters': cl})

# ---------- 計算（元 CSV から独立に再計算） ----------
metrics = []
for r in runs:
    for row in r['rows']:
        if row['status'] != 'LOCATED':
            continue
        xr, yr = float(row['xr']), float(row['yr'])
        psi = math.radians(float(row['psi_deg']))
        cxw, cyw = xr + CAM_X * math.cos(psi), yr + CAM_X * math.sin(psi)
        dx, dy = GT[0] - cxw, GT[1] - cyw
        fwd = dx * math.cos(psi) + dy * math.sin(psi)          # 真値までの光軸方向の距離
        lat = -dx * math.sin(psi) + dy * math.cos(psi)         # 左が正
        phi_true = math.degrees(math.atan2(lat, fwd))
        X, Y = float(row['X']), float(row['Y'])
        metrics.append({
            'run': r['label'], 'csv': f"search_20261024_{r['tag']}.csv", 'cycle': int(row['cycle']),
            'psi_deg': float(row['psi_deg']), 'u_c': float(row['u_c']), 'y2': float(row['y2']),
            'phi_deg': float(row['phi_deg']), 'phi_true_deg': phi_true,
            'dphi_deg': wrap_deg(float(row['phi_deg']) - phi_true),
            'z_a': float(row['z_a']), 'z_true': fwd, 'dz': float(row['z_a']) - fwd,
            'X': X, 'Y': Y, 'dX': X - GT[0], 'dY': Y - GT[1], 'planar': math.hypot(X - GT[0], Y - GT[1]),
        })

groups = {g: [m for m in metrics if m['run'] == g] for g in LOC_RUNS}
summary = {}
for g, ms in groups.items():
    summary[g] = {k: st.fmean(m[k] for m in ms) for k in ('dphi_deg', 'dz', 'planar', 'u_c', 'z_true')}
    summary[g]['n'] = len(ms)

# ---------- 検証（件数・状態・STEP 18-A との一致） ----------
checks = []
total_rows = sum(len(r['rows']) for r in runs)
n_loc = sum(1 for r in runs for row in r['rows'] if row['status'] == 'LOCATED')
n_nop = sum(1 for r in runs for row in r['rows'] if row['status'] == 'NO_PERSON')
checks.append(('全行数 = 34', total_rows == 34, total_rows))
checks.append(('LOCATED = 16', n_loc == 16, n_loc))
checks.append(('NO_PERSON = 18', n_nop == 18, n_nop))
checks.append(('件数 T2=10, S1=3, k=6=3', [summary[g]['n'] for g in LOC_RUNS] == [10, 3, 3],
               [summary[g]['n'] for g in LOC_RUNS]))
ref18a = {'T2': (0.50, -0.108, 0.120), 'S1': (1.16, -0.104, 0.127), 'k=6': (0.11, -0.137, 0.148)}
for g, (a, b, c) in ref18a.items():
    s = summary[g]
    # 作業ログの Δφ は各行を小数 2 桁に丸めてから平均した値（k=6: (0.83+0.16-0.65)/3=0.113 → +0.11）。丸めの差を許す
    ok = abs(s['dphi_deg'] - a) < 0.01 and abs(s['dz'] - b) < 0.0006 and abs(s['planar'] - c) < 0.0006
    checks.append((f'STEP 18-A と一致（{g}、Δφ は ±0.01° まで）', ok, (round(s['dphi_deg'], 3), round(s['dz'], 4), round(s['planar'], 4))))
for r in runs:
    for c in r['clusters']:
        ids = [m for m in metrics if m['run'] == r['label']]
        mx, my = st.fmean(m['X'] for m in ids), st.fmean(m['Y'] for m in ids)
        # CSV の X, Y は小数 4 桁に丸めた値。クラスタ平均は丸める前の値から計算されているので ±1.5e-4 まで許す
        ok = abs(mx - float(c['mean_X'])) < 1.5e-4 and abs(my - float(c['mean_Y'])) < 1.5e-4
        checks.append((f'クラスタ平均の再計算（{r["label"]}）', ok, (round(mx, 4), round(my, 4), c['mean_X'], c['mean_Y'])))
for r in runs:
    for c in r['clusters']:
        empty = all(c[k] == '' for k in ('gt_X', 'gt_Y', 'err_X', 'err_Y', 'err_planar'))
        checks.append((f'クラスタ CSV の gt/err 列が空（{r["label"]}）', empty, ''))

# ---------- 出力：計算表 ----------
fields = ['run', 'csv', 'cycle', 'psi_deg', 'u_c', 'y2', 'phi_deg', 'phi_true_deg', 'dphi_deg',
          'z_a', 'z_true', 'dz', 'X', 'Y', 'dX', 'dY', 'planar']
with open(os.path.join(HERE, 'metrics_step15-17.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for m in metrics:
        w.writerow({k: (f'{m[k]:.4f}' if isinstance(m[k], float) else m[k]) for k in fields})


# ---------- SVG の部品 ----------
def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, s, size=13, fill=INK, anchor='start', weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{esc(s)}</text>')


def marker(shape, x, y, color, r=5, title=''):
    t = f'<title>{esc(title)}</title>' if title else ''
    ring = f'stroke="{SURF}" stroke-width="2"'
    if shape == 'circle':
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" {ring}>{t}</circle>'
    if shape == 'square':
        return f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1.5" fill="{color}" {ring}>{t}</rect>'
    p = f'{x:.1f},{y - r * 1.2:.1f} {x - r * 1.1:.1f},{y + r * 0.8:.1f} {x + r * 1.1:.1f},{y + r * 0.8:.1f}'
    return f'<polygon points="{p}" fill="{color}" {ring}>{t}</polygon>'


def star(x, y, r=9, title=''):
    pts = []
    for i in range(10):
        a = math.radians(-90 + 36 * i)
        rr = r if i % 2 == 0 else r * 0.45
        pts.append(f'{x + rr * math.cos(a):.1f},{y + rr * math.sin(a):.1f}')
    t = f'<title>{esc(title)}</title>' if title else ''
    return f'<polygon points="{" ".join(pts)}" fill="{INK}" stroke="{SURF}" stroke-width="2">{t}</polygon>'


def svg_wrap(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}" role="img" aria-label="{esc(title)}">'
            f'<rect width="{w}" height="{h}" fill="{SURF}"/>{body}</svg>')


# ---------- F1：向きと探索結果 ----------
def make_f1():
    W, H = 980, 1110
    s = 112.0
    rx = st.fmean(float(row['xr']) for r in runs for row in r['rows'])
    ry = st.fmean(float(row['yr']) for r in runs for row in r['rows'])
    ox, oy = 470, 545

    def P(x, y):
        return ox + (x - rx) * s, oy - (y - ry) * s

    b = []
    b.append(text(24, 36, 'F1　一周探索：ロボットの向きと探索結果（STEP 15〜17、9 回の実行）', 19, INK, weight=600))
    b.append(text(24, 60, '扇形＝カメラの幾何学的な水平画角 ±%.1f°（fx=530.467 からの計算値）。検出できる範囲そのものではない' % HALF_FOV, 13, INK2))
    for gx in range(-5, 3):
        x, _ = P(gx, ry)
        if 20 <= x <= W - 20:
            b.append(f'<line x1="{x:.1f}" y1="100" x2="{x:.1f}" y2="{H - 150}" stroke="{GRID}" stroke-width="1"/>')
    for gy in range(-4, 4):
        _, y = P(rx, gy)
        if 100 <= y <= H - 150:
            b.append(f'<line x1="20" y1="{y:.1f}" x2="{W - 20}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
    R = 3.25  # 人物（カメラから約 3.0 m）が扇の内側に入る長さ。扇の長さ自体に意味はない
    for r in runs:
        psi = float(r['rows'][0]['psi_deg'])
        statuses = {row['status'] for row in r['rows']}
        located = statuses == {'LOCATED'}
        a1, a2 = math.radians(psi - HALF_FOV), math.radians(psi + HALF_FOV)
        x1, y1 = P(rx + R * math.cos(a1), ry + R * math.sin(a1))
        x2, y2 = P(rx + R * math.cos(a2), ry + R * math.sin(a2))
        if located:
            style = f'fill="#2a78d6" fill-opacity="0.13" stroke="#2a78d6" stroke-width="2"'
        else:
            style = f'fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 5"'
        n = len(r['rows'])
        cnt = sum(1 for row in r['rows'] if row['status'] == 'LOCATED')
        tip = f"{r['label']}  向き {psi:.2f}°  LOCATED {cnt}/{n}  ({r['desc']})"
        b.append(f'<path d="M{ox},{oy} L{x1:.1f},{y1:.1f} A{R * s:.1f},{R * s:.1f} 0 0 0 {x2:.1f},{y2:.1f} Z" {style}><title>{esc(tip)}</title></path>')
        hx, hy = P(rx + R * math.cos(math.radians(psi)), ry + R * math.sin(math.radians(psi)))
        b.append(f'<line x1="{ox}" y1="{oy}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{"#2a78d6" if located else MUTED}" stroke-width="1.5"/>')
    # ラベル（重なりを避けるため、同じ向きの T2・S1 は1つにまとめる）
    labels = {}
    for r in runs:
        psi = round(float(r['rows'][0]['psi_deg']), 2)
        labels.setdefault(psi, []).append(r)
    for psi, rs in labels.items():
        lx, ly = P(rx + (R + 0.42) * math.cos(math.radians(psi)), ry + (R + 0.42) * math.sin(math.radians(psi)))
        names = '・'.join(x['label'] for x in rs)
        n = sum(len(x['rows']) for x in rs)
        cnt = sum(1 for x in rs for row in x['rows'] if row['status'] == 'LOCATED')
        res = f'LOCATED {cnt}/{n}' if cnt else f'NO_PERSON {n}/{n}'
        anchor = 'middle'
        b.append(text(lx, ly - 4, f'{names}  {psi % 360:.1f}°', 13, INK, anchor, 600))
        b.append(text(lx, ly + 13, res, 12, INK2, anchor))
    # ロボットと真値
    b.append(f'<circle cx="{ox}" cy="{oy}" r="7" fill="{INK2}" stroke="{SURF}" stroke-width="2"><title>ロボット（9 回の xr, yr の平均 ({rx:.4f}, {ry:.4f})。各回の差は最大 約5 mm）</title></circle>')
    b.append(text(ox - 12, oy + 24, 'ロボット', 12, INK2, 'end'))
    gx, gy = P(*GT)
    b.append(star(gx, gy, 10, '人物の真値 (1.0, −0.9)：ワールドファイルの設定値'))
    b.append(text(gx + 14, gy + 20, '人物（真値・設定値）', 12, INK))
    # 凡例と注記
    ly = H - 110
    b.append(f'<rect x="24" y="{ly - 10}" width="22" height="14" fill="#2a78d6" fill-opacity="0.13" stroke="#2a78d6" stroke-width="2"/>')
    b.append(text(54, ly + 2, 'LOCATED（人物を検出し座標を推定）', 13))
    b.append(f'<rect x="330" y="{ly - 10}" width="22" height="14" fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 5"/>')
    b.append(text(360, ly + 2, 'NO_PERSON（検出なし）', 13))
    b.append(text(24, ly + 30, '注：検出できた向きは 0° と 約8.9° の 2 種類だけ。検出の境界（画角の端付近）は測定していない。', 12, INK2))
    b.append(text(24, ly + 48, '　　k=5（316.0°）では人物はカメラから計算上 約+37° の位置（画角の外）。上から見た図・ワールド座標（+x 右、+y 上）、1 マス 1 m。扇の長さ（3.25 m）に意味はない。', 12, INK2))
    b.append(text(24, ly + 66, '出典：results/step15-17 の 9 つの CSV（psi_deg・xr・yr・status）。真値はワールドファイル、fx は camera_info の値で、CSV には含まれない。', 12, INK2))
    return svg_wrap(W, H, ''.join(b), 'F1 向きと探索結果')


# ---------- F2：推定位置と真値 ----------
def make_f2():
    W, H = 980, 640
    x0, x1, y0, y1 = 0.80, 1.04, -0.96, -0.80
    pl, pt = 90, 100
    s = 560 / (x1 - x0)                      # 縦横同じ縮尺
    pr, pb = pl + (x1 - x0) * s, pt + (y1 - y0) * s

    def P(x, y):
        return pl + (x - x0) * s, pb - (y - y0) * s

    b = [text(24, 36, 'F2　推定した人物の位置と真値（LOCATED 16 件）', 19, INK, weight=600),
         text(24, 60, '条件：人物 1 人・配置 1 つ・距離 約2.9 m・検出時のロボットの向き 0° と 約8.9°（シミュレーション）', 13, INK2)]
    v = x0
    while v <= x1 + 1e-9:
        X, _ = P(v, y0)
        b.append(f'<line x1="{X:.1f}" y1="{pt}" x2="{X:.1f}" y2="{pb}" stroke="{GRID}"/>')
        b.append(text(X, pb + 18, f'{v:.2f}', 11, MUTED, 'middle'))
        v += 0.04
    v = y0
    while v <= y1 + 1e-9:
        _, Y = P(x0, v)
        b.append(f'<line x1="{pl}" y1="{Y:.1f}" x2="{pr:.1f}" y2="{Y:.1f}" stroke="{GRID}"/>')
        b.append(text(pl - 8, Y + 4, f'{v:.2f}', 11, MUTED, 'end'))
        v += 0.04
    b.append(f'<rect x="{pl}" y="{pt}" width="{pr - pl:.1f}" height="{pb - pt}" fill="none" stroke="{AXIS}"/>')
    b.append(text((pl + pr) / 2, pb + 40, 'ワールド座標 X [m]', 13, INK2, 'middle'))
    b.append(f'<text x="{pl - 56}" y="{(pt + pb) / 2:.1f}" font-size="13" fill="{INK2}" text-anchor="middle" transform="rotate(-90 {pl - 56} {(pt + pb) / 2:.1f})">ワールド座標 Y [m]</text>')
    # ロボットの方向（図の外）
    ax, ay = P(0.81, -0.952)
    b.append(f'<line x1="{ax + 50:.1f}" y1="{ay:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="{INK2}" stroke-width="1.5" marker-end="url(#arr)"/>')
    b.append(text(ax + 58, ay + 4, 'ロボットはこの方向 約3 m 先 (−2.0, −0.5)', 11, INK2))
    b.insert(0, f'<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="{INK2}"/></marker></defs>')
    # 生の推定点
    for m in metrics:
        color, shape = SERIES[m['run']]
        X, Y = P(m['X'], m['Y'])
        b.append(marker(shape, X, Y, color, 5, f"{m['run']} 周{m['cycle']}: ({m['X']:.4f}, {m['Y']:.4f})  平面誤差 {m['planar']:.3f} m"))
    # クラスタ平均と標準偏差
    for r in runs:
        if r['label'] not in LOC_RUNS:
            continue
        c = r['clusters'][0]
        mx, my, sx, sy = (float(c[k]) for k in ('mean_X', 'mean_Y', 'std_X', 'std_Y'))
        color, shape = SERIES[r['label']]
        X, Y = P(mx, my)
        b.append(f'<line x1="{X - sx * s:.1f}" y1="{Y:.1f}" x2="{X + sx * s:.1f}" y2="{Y:.1f}" stroke="{color}" stroke-width="2"/>')
        b.append(f'<line x1="{X:.1f}" y1="{Y - sy * s:.1f}" x2="{X:.1f}" y2="{Y + sy * s:.1f}" stroke="{color}" stroke-width="2"/>')
        b.append(marker(shape, X, Y, color, 9, f"{r['label']} クラスタ平均 ({mx:.4f}, {my:.4f})  標準偏差 X {sx:.4f} / Y {sy:.4f}  n={c['n']}"))
        off = {'T2': (14, 4), 'S1': (14, 4), 'k=6': (-14, 4)}[r['label']]
        b.append(text(X + off[0], Y + off[1], f"{r['label']} 平均", 12, INK, 'end' if off[0] < 0 else 'start', 600))
    gx, gy = P(*GT)
    b.append(star(gx, gy, 11, '真値 (1.0, −0.9)：ワールドファイルのアクター基準点（設定値）'))
    b.append(text(gx, gy - 18, '真値 (1.0, −0.9)', 12, INK, 'middle', 600))
    # 凡例
    lx, ly = pr + 30, 110
    b.append(text(lx, ly, '凡例', 13, INK, weight=600))
    rows_leg = [('T2', 'STEP 15 T2・向き 0°（n=10）'), ('S1', 'STEP 16 S1・向き 0°（n=3）'), ('k=6', 'STEP 17 k=6・向き 約8.9°（n=3）')]
    for i, (g, lab) in enumerate(rows_leg):
        color, shape = SERIES[g]
        y = ly + 26 + i * 24
        b.append(marker(shape, lx + 8, y - 4, color, 5))
        b.append(text(lx + 22, y, lab, 12, INK))
    y = ly + 26 + 3 * 24 + 6
    b.append(text(lx, y, '小さい印：1 周ごとの推定値（CSV の X, Y）', 12, INK2))
    b.append(text(lx, y + 18, '大きい印＋線：クラスタ平均 ± 標準偏差', 12, INK2))
    b.append(text(lx, y + 36, '（座標ごとの標本標準偏差。標準誤差や', 12, INK2))
    b.append(text(lx, y + 54, '　信頼区間ではない。クラスタ CSV の値）', 12, INK2))
    b.append(star(lx + 8, y + 82, 8))
    b.append(text(lx + 22, y + 86, '真値：ワールドファイルの設定値', 12, INK))
    b.append(text(lx, y + 104, '（CSV に記録された実測値ではない）', 12, INK2))
    b.append(text(24, H - 70, '平面誤差（クラスタ平均と真値の距離、計算値）：' + '、'.join(
        f"{g} {math.hypot(float(r['clusters'][0]['mean_X']) - GT[0], float(r['clusters'][0]['mean_Y']) - GT[1]):.3f} m"
        for r in runs for g in [r['label']] if g in LOC_RUNS) + '（F3 の 1 周ごとの平面誤差の平均とは別の量）', 12, INK2))
    b.append(text(24, H - 50, '推定値は真値よりロボット側（手前）に約 0.1〜0.15 m まとまっている。補正値は適用していない。今回の条件だけの結果で、一般的な精度の保証ではない。', 12, INK2))
    b.append(text(24, H - 30, '出典：results/step15-17 の LOCATED 行（X, Y）とクラスタ CSV（mean_X/Y, std_X/Y）。縦横は同じ縮尺。', 12, INK2))
    return svg_wrap(W, H, ''.join(b), 'F2 推定位置と真値')


# ---------- F3：誤差の内訳 ----------
def make_f3():
    W, H = 1000, 640
    panels = [('dz', 'ΔZ：距離Aの誤差 [m]', -0.16, 0.02, 0.04, '{:.2f}'),
              ('dphi_deg', 'Δφ：方向の誤差 [°]', -1.5, 1.5, 0.5, '{:.1f}'),
              ('planar', '平面誤差 [m]', 0.0, 0.18, 0.03, '{:.2f}')]
    b = [text(24, 36, 'F3　位置推定の誤差の内訳（計算値、LOCATED 16 件）', 19, INK, weight=600),
         text(24, 60, '各値 ＝ 推定値 − 真値（真値はワールドファイルの設定値と各行の odom から計算）。点＝1 周ごと、横線＝実行ごとの平均', 13, INK2)]
    pw, gap, top, bot = 240, 80, 110, 470
    for i, (key, title, lo, hi, step, fmtv) in enumerate(panels):
        x0 = 70 + i * (pw + gap)

        def Y(v):
            return bot - (v - lo) / (hi - lo) * (bot - top)
        b.append(text(x0, top - 14, title, 14, INK, weight=600))
        v = lo
        while v <= hi + 1e-9:
            y = Y(v)
            zero = abs(v) < 1e-9
            b.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + pw}" y2="{y:.1f}" stroke="{AXIS if zero else GRID}" stroke-width="{1.5 if zero else 1}"/>')
            b.append(text(x0 - 6, y + 4, fmtv.format(v), 11, MUTED, 'end'))
            v += step
        for j, g in enumerate(LOC_RUNS):
            cxp = x0 + (j + 0.5) * pw / 3
            ms = groups[g]
            color, shape = SERIES[g]
            for k, m in enumerate(ms):
                jitter = ((k % 5) - 2) * 6 if len(ms) > 3 else (k - 1) * 8
                b.append(marker(shape, cxp + jitter, Y(m[key]), color, 4.5,
                                f"{g} 周{m['cycle']}: {key} = {m[key]:.4f}"))
            mean = st.fmean(m[key] for m in ms)
            b.append(f'<line x1="{cxp - 32:.1f}" y1="{Y(mean):.1f}" x2="{cxp + 32:.1f}" y2="{Y(mean):.1f}" stroke="{INK}" stroke-width="2"><title>{g} 平均 {mean:.4f}</title></line>')
            b.append(text(cxp + 34, Y(mean) + 4, f'{mean:+.2f}' if key == 'dphi_deg' else f'{mean:.3f}', 11, INK))
            b.append(text(cxp, bot + 20, g, 12, INK, 'middle', 600))
            b.append(text(cxp, bot + 36, f'n={len(ms)}', 11, INK2, 'middle'))
    b.append(text(24, H - 104, f'・ΔZ（距離Aの誤差）は 16 件すべて負で、推定距離は真値より手前（{max(m["dz"] for m in metrics):+.3f}〜{min(m["dz"] for m in metrics):+.3f} m）。ばらつきは小さく、系統的な偏りの可能性が高い。', 12, INK2))
    b.append(text(24, H - 84, '　原因（つま先側の接地点、YOLO の枠の下端、斜めからの見え方など）は未検証の仮説で、ここでは断定しない。補正値は適用していない。', 12, INK2))
    b.append(text(24, H - 64, f'・Δφ（方向の誤差）は {min(m["dphi_deg"] for m in metrics):+.2f}〜{max(m["dphi_deg"] for m in metrics):+.2f}°。同じ向きでも枠の中心が揺れる（アクターのアニメーションと推定、未確認）。', 12, INK2))
    b.append(text(24, H - 44, '・件数が少なく（10・3・3）、条件は 1 配置だけ。距離の定義：カメラ（base_footprint から前方 0.08 m）から見た光軸方向の距離。', 12, INK2))
    b.append(text(24, H - 24, '出典：results/step15-17 の LOCATED 行（phi_deg, z_a, X, Y, xr, yr, psi_deg）から再計算。計算表：metrics_step15-17.csv', 12, INK2))
    return svg_wrap(W, H, ''.join(b), 'F3 誤差の内訳')


figs = {'f1_headings.svg': make_f1(), 'f2_positions.svg': make_f2(), 'f3_errors.svg': make_f3()}
for name, svg in figs.items():
    with open(os.path.join(HERE, name), 'w', encoding='utf-8') as f:
        f.write(svg)

# ---------- HTML（3 枚と計算表） ----------
rows_html = ''.join(
    f"<tr><td>{m['run']}</td><td>{m['cycle']}</td><td>{m['psi_deg']:.2f}</td><td>{m['u_c']:.1f}</td>"
    f"<td>{m['phi_deg']:.3f}</td><td>{m['phi_true_deg']:.3f}</td><td>{m['dphi_deg']:+.3f}</td>"
    f"<td>{m['z_a']:.4f}</td><td>{m['z_true']:.4f}</td><td>{m['dz']:+.4f}</td>"
    f"<td>{m['X']:.4f}</td><td>{m['Y']:.4f}</td><td>{m['planar']:.4f}</td></tr>" for m in metrics)
html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>STEP 15-17 可視化</title><style>
:root{{color-scheme:light}} body{{margin:0;padding:16px;background:#f9f9f7;color:{INK};font-family:{FONT}}}
figure{{margin:0 0 28px;overflow-x:auto}} svg{{max-width:100%;height:auto;border:1px solid rgba(11,11,11,.1);border-radius:6px}}
table{{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}} td,th{{border-bottom:1px solid {GRID};padding:4px 8px;text-align:right}}
th{{color:{INK2};font-weight:600}} .wrap{{overflow-x:auto}}
</style></head><body>
<h1 style="font-size:20px">STEP 15〜17 探索・位置推定の可視化</h1>
<p style="color:{INK2}">点や扇形にカーソルを合わせると値が表示されます。値の種類（実測・設定値・計算値）と限界は README.md を参照。</p>
{''.join(f'<figure>{svg}</figure>' for svg in figs.values())}
<h2 style="font-size:16px">計算表（LOCATED 16 件、metrics_step15-17.csv と同じ値）</h2>
<div class="wrap"><table><thead><tr><th>実行</th><th>周</th><th>向き°</th><th>u_c</th><th>φ推定°</th><th>φ真値°</th><th>Δφ°</th>
<th>Z推定 m</th><th>Z真値 m</th><th>ΔZ m</th><th>X</th><th>Y</th><th>平面誤差 m</th></tr></thead><tbody>{rows_html}</tbody></table></div>
</body></html>"""
with open(os.path.join(HERE, 'figures.html'), 'w', encoding='utf-8') as f:
    f.write(html)

# ---------- 検証結果の表示 ----------
print(f'half FOV (計算値) = {HALF_FOV:.3f} deg')
for g in LOC_RUNS:
    s_ = summary[g]
    print(f"{g}: n={s_['n']} mean dphi={s_['dphi_deg']:+.3f} deg, dz={s_['dz']:+.4f} m, planar={s_['planar']:.4f} m, "
          f"u_c={s_['u_c']:.1f}, z_true={s_['z_true']:.3f}")
bad = 0
for name, ok, val in checks:
    print(f"[{'OK' if ok else 'NG'}] {name} {val}")
    bad += (not ok)
print('CHECKS_FAILED =', bad)
