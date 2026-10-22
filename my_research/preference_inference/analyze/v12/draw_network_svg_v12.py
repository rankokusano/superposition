"""
Thesis figure: the v11 and v12 networks as a clean SVG in the style of docs/figs/research_overview_network.svg
(the paper's network). Writes docs/figs/network_v11_v12.svg. No data are read.
Colours: blue = trained in stage 1 (frozen in stage 2), orange = trained in stage 2 (part G only),
hatched grey = frozen v4 weights (v12 encoders). Red text = inputs to the shared module.
"""

import os

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(_PI, 'docs', 'figs', 'network_v11_v12.svg')
BLUE, ORANGE, GREY, RED, DARK = '#dbe8f5', '#fde2c4', '#f2f2f2', '#b3261e', '#1f5f9f'


def box(x, y, w, h, fill, lines, sizes=None, hatch=False):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="#333"/>']
    if hatch:
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="url(#hatch)" stroke="#333"/>')
    n = len(lines)
    for i, t in enumerate(lines):
        fs = (sizes or [13] + [11] * 5)[i]
        col = '#111' if i == 0 else '#555'
        yy = y + h / 2 + (i - (n - 1) / 2) * 17 + 4
        s.append(f'<text x="{x + w / 2}" y="{yy:.0f}" text-anchor="middle" font-size="{fs}" fill="{col}">{t}</text>')
    return '\n'.join(s)


def arrow(x1, y1, x2, y2, col='#333', w=1.2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{w}" marker-end="url(#ar)"/>'


def text(x, y, t, size=11, col='#555', anchor='start', weight='normal'):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{col}" font-weight="{weight}">{t}</text>'


def panel(ox, version):
    v12 = version == 'v12'
    s = [f'<g transform="translate({ox},0)">']
    title = ('v11：快・不快だけを入力し、全体を一から学習' if not v12 else
             'v12：運動を戻し、ときどき隠す（Encoder は v4 で凍結）')
    s.append(text(345, 74, title, 15, '#111', 'middle', 'bold'))
    # process-1 input (top), centred short lines
    if v12:
        lines = [('運動 m_t（2）＋ 快・不快 P_t（8）＋ 隠れの印（1）', 12),
                 ('1 段目：運動を 5〜15 ステップの区間で隠し、隠れた間は P_t を保つ', 10),
                 ('2 段目：運動は隠さない', 10)]
    else:
        lines = [('快・不快 P_t（8）：方向ごとの（赤 − 緑）の画素', 12), ('運動は入れない', 10)]
    for i, (t, fs) in enumerate(lines):
        s.append(text(440, 98 + 15 * i, t, fs, RED, 'middle'))
    s.append(arrow(440, 98 + 15 * len(lines) - 6, 440, 179, RED, 1.6))
    # input view
    s.append(box(10, 285, 120, 56, GREY, ['A-1 の視界 v_t', '16×64 の全方位'], [13, 11]))
    # encoders
    enc_fill = GREY if v12 else BLUE
    sub = 'v4 の重みで凍結' if v12 else '1 段目で学習'
    s.append(box(170, 180, 120, 50, enc_fill, ['Encoder-1', sub], [13, 11], hatch=v12))
    s.append(box(170, 400, 120, 50, enc_fill, ['Encoder-2', sub], [13, 11], hatch=v12))
    s.append(arrow(130, 302, 169, 208)); s.append(arrow(130, 326, 169, 422))
    # shared module
    s.append(f'<rect x="335" y="160" width="210" height="330" rx="10" fill="none" stroke="{DARK}" stroke-dasharray="6 4"/>')
    s.append(text(440, 312, '共有モジュール（LSTM 128）', 12, DARK, 'middle'))
    s.append(text(440, 329, '重みは共通・記憶は別', 11, DARK, 'middle'))
    s.append(box(355, 180, 170, 56, BLUE, ['process-1（自己）', '内部状態 h¹'], [13, 12]))
    s.append(box(355, 412, 170, 56, BLUE, ['process-2（他者）', '内部状態 h²'], [13, 12]))
    s.append(arrow(290, 205, 354, 208)); s.append(arrow(290, 425, 354, 435))
    s.append(text(320, 197, 'マスク', 10, RED, 'middle')); s.append(text(320, 417, 'マスク', 10, RED, 'middle'))
    # process-2 input (bottom)
    s.append(arrow(440, 560, 440, 469, RED, 1.6))
    s.append(text(452, 520, '1 段目：零', 12, RED))
    s.append(text(452, 536, '2 段目：' + ('[0, 0, Q2, 1]' if v12 else 'Q2'), 12, RED))
    s.append(text(452, 551, ('（運動 0・快・不快の枠に Q2・印）' if v12 else '（快・不快の枠に Q2）'), 10, RED))
    # part G
    s.append(box(120, 528, 220, 62, ORANGE, ['部品 G', 'Motion Generator と同じ作り', '2 段目でこれだけ学習 → Q2（8）'], [13, 11, 11]))
    s.append(arrow(225, 451, 228, 527)); s.append(text(120, 492, 'マスク前の出力', 10))
    s.append('<polyline points="340,559 395,559 432,472" fill="none" stroke="#333" marker-end="url(#ar)"/>')
    if v12:
        s.append(text(10, 612, '2 段目 b：Q2 を 1 段目の快・不快の形に限る（範囲 2〜5 は 0、範囲 0・1 と 6・7 は逆の符号）', 11, '#8a4b00'))
        s.append(text(10, 628, '2 段目 c：さらに、入力の視界から A-2 を 10〜30 ステップの区間で消す（正解の視界はそのまま）', 11, '#8a4b00'))
    # integration, decoder, prediction, feature prediction
    s.append(box(575, 290, 95, 50, BLUE, ['統合部'], [13]))
    s.append(arrow(525, 212, 574, 302)); s.append(arrow(525, 438, 574, 330))
    s.append(box(575, 365, 95, 50, BLUE, ['Decoder'], [13]))
    s.append(arrow(622, 340, 622, 364))
    s.append(box(565, 440, 115, 42, GREY, ['予測 v_{t+1}'], [13]))
    s.append(arrow(622, 415, 622, 439))
    s.append(box(565, 180, 115, 56, BLUE, ['特徴予測部', '次の Encoder 出力'], [13, 11]))
    s.append(arrow(525, 200, 564, 205))
    s.append('</g>')
    return '\n'.join(s)


def main():
    W, H = 1420, 740
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         'font-family="Hiragino Sans, Noto Sans JP, sans-serif" font-size="13">',
         '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker>'
         '<pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<line x1="0" y1="0" x2="0" y2="8" stroke="#c4c4c4" stroke-width="2"/></pattern></defs>',
         f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
         text(W / 2, 26, 'v11・v12 のネットワーク（元論文の重ね合わせのネットワークに、快・不快の入力と部品 G を加えたもの）', 16, '#111', 'middle', 'bold'),
         text(W / 2, 46, '予測の対象は A-1 自身の次の視界だけ。1 段目は A-1 が経験あり（半分は赤の方へ）／なし（ランダム）で学習し、2 段目は部品 G だけを学習する', 12, '#555', 'middle'),
         panel(10, 'v11'), panel(720, 'v12'),
         f'<line x1="712" y1="60" x2="712" y2="640" stroke="#ccc"/>']
    # legend
    y = 668
    s.append(f'<rect x="20" y="{y}" width="18" height="14" fill="{BLUE}" stroke="#333"/>')
    s.append(text(44, y + 12, '1 段目で学習（2 段目は凍結）', 12, '#333'))
    s.append(f'<rect x="270" y="{y}" width="18" height="14" fill="{ORANGE}" stroke="#333"/>')
    s.append(text(294, y + 12, '2 段目で学習（部品 G だけ）', 12, '#333'))
    s.append(f'<rect x="520" y="{y}" width="18" height="14" fill="{GREY}" stroke="#333"/>')
    s.append(f'<rect x="520" y="{y}" width="18" height="14" fill="url(#hatch)" stroke="#333"/>')
    s.append(text(544, y + 12, '凍結した v4 の重み（元論文の exp1 の再現、v12 の Encoder）', 12, '#333'))
    s.append(text(20, y + 38, '赤字：共有モジュールへの入力。マスク：1 段目は学習中 99% の確率で視覚の符号を零にする（元論文と同じ）。2 段目は t > 0 で常に零（元論文の exp3 と同じ）。', 12, '#333'))
    s.append(text(20, y + 56, 'Fig.5：Encoder-1 → Decoder の Autoencoder を学習し、Encoder-2 の出力を同じ Decoder に通して「A-2 から見た視界」を取り出す。', 12, '#333'))
    s.append('</svg>')
    open(OUT, 'w').write('\n'.join(s))
    print('saved', OUT)


if __name__ == '__main__':
    main()
