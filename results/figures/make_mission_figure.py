#!/usr/bin/env python3
"""make_mission_figure.py : デモ本番（mission_runner.py --run --allow-spin）の記録から F4 を作る。

入力（読むだけ）: results/step19/mission_20261025_081118/（動画に映っている実行。各ラウンドの CSV、report_06.json）
出力（このフォルダ）: f4_mission.svg（PNG は f1〜f3 と同じく Edge のヘッドレスモードで画像化）
真値 (1.0, -0.9) はワールドファイルの設定値、水平画角 ±31.1° は fx=530.467・幅640 からの計算値で、どちらも CSV にはない。
"""
import csv
import glob
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
MISSION_NAME = 'mission_20261025_081118'
MISSION = os.path.normpath(os.path.join(HERE, '..', 'step19', MISSION_NAME))
GT = (1.0, -0.9)
HALF_FOV = math.degrees(math.atan(320 / 530.467))
INK, INK2, MUTED, GRID, SURF, BLUE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb', '#2a78d6'
FONT = "'Yu Gothic UI','Meiryo','Segoe UI',system-ui,sans-serif"


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, s, size=13, fill=INK, anchor='start', weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{esc(s)}</text>')


rounds = []
for f in sorted(glob.glob(os.path.join(MISSION, 'r0*', 'search_*[0-9].csv'))):
    with open(f, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    rounds.append({'id': os.path.basename(os.path.dirname(f)), 'rows': rows,
                   'psi': float(rows[0]['psi_deg']), 'xr': float(rows[0]['xr']), 'yr': float(rows[0]['yr'])})
with open(os.path.join(MISSION, 'report_06.json'), encoding='utf-8') as fh:
    rep = json.load(fh)
cl = rep['clusters'][0]
mx, my = cl['mean_XY']
err = math.hypot(mx - GT[0], my - GT[1])

W, H = 1000, 820
s, R = 85.0, 3.25
rx, ry = st.fmean(r['xr'] for r in rounds), st.fmean(r['yr'] for r in rounds)
ox, oy = 350, 430


def P(x, y):
    return ox + (x - rx) * s, oy - (y - ry) * s


b = [text(24, 36, 'F4　デモ：旋回しながら探索し、人物を見つけて報告する（mission_runner.py --run --allow-spin）', 18, INK, weight=600),
     text(24, 60, f'開始 {rounds[0]["psi"] % 360:.1f}° → 承認つきの旋回 {sum(1 for a, c in zip(rounds, rounds[1:]) if abs(c["psi"] - a["psi"]) > 1)} 回 → '
                  f'{rounds[-1]["psi"] % 360:.1f}° で検出 → 同じ向きで再観測 → REPORTED（扇形＝幾何学的な画角 ±{HALF_FOV:.1f}°）', 13, INK2)]
for gx in range(-6, 3):
    x, _ = P(gx, ry)
    if 20 <= x <= 700:
        b.append(f'<line x1="{x:.1f}" y1="80" x2="{x:.1f}" y2="{H - 60}" stroke="{GRID}"/>')
for gy in range(-4, 4):
    _, y = P(rx, gy)
    if 80 <= y <= H - 60:
        b.append(f'<line x1="20" y1="{y:.1f}" x2="700" y2="{y:.1f}" stroke="{GRID}"/>')
seen = {}
for i, r in enumerate(rounds):
    psi = r['psi']
    found = any(row['status'] == 'LOCATED' for row in r['rows'])
    key = round(psi, 1)
    if key in seen:                      # 同じ向きの再観測（r06）は扇形を重ねない
        seen[key].append(r['id'])
        continue
    seen[key] = [r['id']]
    a1, a2 = math.radians(psi - HALF_FOV), math.radians(psi + HALF_FOV)
    x1, y1 = P(rx + R * math.cos(a1), ry + R * math.sin(a1))
    x2, y2 = P(rx + R * math.cos(a2), ry + R * math.sin(a2))
    style = (f'fill="{BLUE}" fill-opacity="0.15" stroke="{BLUE}" stroke-width="2"' if found
             else f'fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 5"')
    b.append(f'<path d="M{ox},{oy} L{x1:.1f},{y1:.1f} A{R * s:.1f},{R * s:.1f} 0 0 0 {x2:.1f},{y2:.1f} Z" {style}/>')
for key, ids in seen.items():
    psi = key
    lx, ly = P(rx + (R + 0.55) * math.cos(math.radians(psi)), ry + (R + 0.5) * math.sin(math.radians(psi)))
    rr = [r for r in rounds if r['id'] in ids]
    res = ' / '.join(f"{r['id']} {sum(1 for x in r['rows'] if x['status'] == 'LOCATED')}/3" for r in rr)
    b.append(text(lx, ly - 4, f'{psi % 360:.1f}°', 13, INK, 'middle', 600))
    b.append(text(lx, ly + 13, res, 11, INK2, 'middle'))
# 旋回の順番（開始 → 終了）を弧の矢印で示す
arc = []
start, end = rounds[0]['psi'], rounds[-1]['psi'] + 360
for k in range(0, 41):
    a = math.radians(start + (end - start) * k / 40)
    x, y = P(rx + 0.75 * math.cos(a), ry + 0.75 * math.sin(a))
    arc.append(f'{x:.1f},{y:.1f}')
b.insert(0, f'<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="{INK2}"/></marker></defs>')
b.append(f'<polyline points="{" ".join(arc)}" fill="none" stroke="{INK2}" stroke-width="1.5" marker-end="url(#ar)"/>')
b.append(f'<circle cx="{ox}" cy="{oy}" r="7" fill="{INK2}" stroke="{SURF}" stroke-width="2"/>')
b.append(text(ox - 10, oy + 26, 'ロボット（その場で左回り）', 11, INK2, 'end'))
for r in rounds:
    for row in r['rows']:
        if row['status'] == 'LOCATED':
            x, y = P(float(row['X']), float(row['Y']))
            b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{BLUE}" stroke="{SURF}" stroke-width="1"/>')
gx, gy = P(*GT)
pts = []
for i in range(10):
    a = math.radians(-90 + 36 * i)
    rr_ = 9 if i % 2 == 0 else 4
    pts.append(f'{gx + rr_ * math.cos(a):.1f},{gy + rr_ * math.sin(a):.1f}')
b.append(f'<polygon points="{" ".join(pts)}" fill="{INK}" stroke="{SURF}" stroke-width="1.5"/>')
b.append(text(gx + 14, gy + 22, '人物（真値・設定値）', 11, INK))

# 右側：ラウンドごとの経過
tx, ty = 730, 120
b.append(text(tx, ty, 'ラウンドごとの経過', 14, INK, weight=600))
y = ty + 26
for r in rounds:
    n = sum(1 for x in r['rows'] if x['status'] == 'LOCATED')
    status = f'LOCATED {n}/3' if n else 'NO_PERSON 3/3'
    b.append(text(tx, y, f"{r['id']}  {r['psi'] % 360:6.1f}°  {status}", 12, INK if n else INK2))
    y += 20
y += 8
for line in ['r01〜r04：見えない → 承認（y）→ 旋回', '　　　　 → 停止確認 → 次の向き',
             'r05：検出（候補）→ 同じ向きで再観測', 'r06：同じ位置で再検出 → REPORTED', '',
             f'報告した位置（平均）({mx:.3f}, {my:.3f})', f'標準偏差 ({cl["std_XY"][0]:.3f}, {cl["std_XY"][1]:.3f}) m',
             f'真値 (1.0, −0.9) との差 {err:.3f} m（計算値）', '旋回 4 回、すべて DONE・静止を確認',
             '終了状態 END_REPORTED（終了コード 0）']:
    b.append(text(tx, y, line, 12, INK2 if line.startswith(('r0', '　')) else INK))
    y += 19
b.append(text(24, H - 34, 'REPORTED は「シミュレーション上の確認条件を満たした」という意味で、現実の発見を保証しない。NO_PERSON は「その観測で検出されなかった」だけ。', 11, INK2))
b.append(text(24, H - 16, f'出典：results/step19/{MISSION_NAME}（各ラウンドの CSV と report_06.json）。上から見た図・ワールド座標、1マス 1 m。', 11, INK2))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">'
       f'<rect width="{W}" height="{H}" fill="{SURF}"/>{"".join(b)}</svg>')
with open(os.path.join(HERE, 'f4_mission.svg'), 'w', encoding='utf-8') as f:
    f.write(svg)
print('rounds', [(r['id'], round(r['psi'], 3)) for r in rounds], 'mean', (mx, my), 'err', round(err, 4))
