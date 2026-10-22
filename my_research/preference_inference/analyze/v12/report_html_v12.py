"""
v11/v12 review report (docs/v12_experiment_log.md Sec.42), CPU only. Collects existing figures (relative paths) with one
note each (what it shows, how to read it, visually checked or not) and the stage-1/2 tables; the post-hoc B table is
filled from whatever data/result/v12/post/*.json exist, so re-running after more pairs finish updates it.
Writes data/result/v12/report/index.html (open it from that folder; images are referenced relatively).
"""

import glob
import html
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v12 import RES  # noqa

OUT = os.path.join(RES, 'v12', 'report')
V11, V12 = '../../v11', '..'


def fig(path, what, read, checked, alt=''):
    ok = os.path.exists(os.path.join(OUT, path))
    mark = '<span class="ok">目視：済み</span>' if checked else '<span class="ng">目視：未確認</span>'
    img = f'<a href="{path}"><img src="{path}" alt="{html.escape(alt or what)}"></a>' if ok \
        else f'<p class="ng">ファイルがない：{html.escape(path)}</p>'
    return (f'<figure>{img}<figcaption><b>{what}</b>　{read}　{mark}'
            f'<br><code>{html.escape(path)}</code></figcaption></figure>')


def links(paths, note='目視：未確認'):
    items = ' ・ '.join(f'<a href="{p}">{html.escape(os.path.basename(p))}</a>' for p in paths
                        if os.path.exists(os.path.join(OUT, p)))
    return f'<p class="more">ほかの組・条件（{note}）：{items}</p>' if items else ''


def f3(x):
    return f'{x:.2f}'


def ci(d):
    lo, hi = d['ci95']
    star = '' if lo <= 0 <= hi else ' *'
    return f'{d["mean"]:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}'


def table_b():
    rows_r, rows_b3 = [], []
    files = sorted(glob.glob(os.path.join(RES, 'v12', 'post', 'v12_s2*_s[0-9].json')))
    for f in files:
        r = json.load(open(f))
        fam = '正式' if r['prefix'] == 'v12_s2' else '2b'
        for st in ('green', 'pref'):
            for k, tl in (('green', '①配置'), ('pref', '②好み')):
                for feat in ('Q2', 'h2'):
                    v = [r[f'{st}_{c}'][f'{k}_{feat}'] for c in ('random', 'mixed')]
                    cells = ''.join(
                        f'<td>{f3(a[key])} / {f3(b_[key])}</td>'
                        for key in ('frames_linear', 'frames_knn', 'pos_only_linear', 'pos_plus_linear', 'episode_linear')
                        for a, b_ in [v])
                    rows_r.append(f'<tr><td>{fam}</td><td>s{r["seed"]}</td><td>{st}</td><td>{tl}</td><td>{feat}</td>{cells}</tr>')
            for k, tl in (('green', '①'), ('pref', '②')):
                p = r['paired_B3'][f'{st}_{k}_freeze']
                mx = r[f'{st}_mixed'][f'{k}_B3_freeze']
                rn = r[f'{st}_random'][f'{k}_B3_freeze']
                rows_b3.append(f'<tr><td>{fam}</td><td>s{r["seed"]}</td><td>{st}</td><td>{tl}</td>'
                               f'<td>{ci(rn["g_other"])}</td><td>{ci(mx["g_other"])}</td><td>{ci(p["g_other"])}</td>'
                               f'<td>{ci(p["g_other_bg"])}</td><td>{f3(rn["frac_read"])} / {f3(mx["frac_read"])}</td></tr>')
    done = ', '.join(os.path.basename(f)[:-5] for f in files)
    head_r = ('<tr><th>段</th><th>組</th><th>部品 G の学習データ</th><th>読む対象</th><th>特徴</th>'
              '<th>フレーム・線形</th><th>フレーム・25 近傍</th><th>位置だけ（2 次）</th><th>位置 ＋ 特徴</th>'
              '<th>エピソード・線形</th></tr>')
    head_b3 = ('<tr><th>段</th><th>組</th><th>部品 G の学習データ</th><th>評価のセット</th><th>経験なし g_other</th>'
               '<th>経験あり g_other</th><th>組の差（主の読み取り）</th><th>組の差（B1 の読み取り）</th><th>読めた割合 なし / あり</th></tr>')
    return (f'<p>計算が終わった分：{done}（残りは終わり次第このページを作り直す）。</p>'
            '<h4>B1・B2：A-2 の好みの読み出しの正答率（経験なし / 経験あり。偶然 0.5。エピソード単位の 5 分割）</h4>'
            f'<div class="scroll"><table>{head_r}{"".join(rows_r)}</table></div>'
            '<h4>B3：生成した Q2 を t=20 で止めたときの A-2 の積み上がり（K=15、平均 [95% CI]、* は CI が 0 を含まない）</h4>'
            f'<div class="scroll"><table>{head_b3}{"".join(rows_b3)}</table></div>')


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--ok:#1a7f37;--ng:#b3261e;--card:#f6f7f9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16181c;--fg:#e8e8ea;--mut:#a0a3a8;--line:#33363c;--ok:#5ec27a;--ng:#ff8a80;--card:#1f2228}}
:root[data-theme="dark"]{--bg:#16181c;--fg:#e8e8ea;--mut:#a0a3a8;--line:#33363c;--ok:#5ec27a;--ng:#ff8a80;--card:#1f2228}
body{background:var(--bg);color:var(--fg);font-family:"Hiragino Sans","Noto Sans JP",sans-serif;line-height:1.7;margin:0 auto;max-width:1200px;padding:16px}
h1{font-size:1.5rem}h2{border-bottom:2px solid var(--line);padding-bottom:4px;margin-top:2.2em}h3{margin-top:1.6em}
figure{margin:12px 0 22px;background:var(--card);padding:10px;border-radius:6px}
figure img{max-width:100%;height:auto;display:block;background:#fff}
figcaption{font-size:.9rem;margin-top:6px}figcaption code{color:var(--mut);font-size:.8rem}
.ok{color:var(--ok);font-weight:bold}.ng{color:var(--ng);font-weight:bold}.more{font-size:.85rem;color:var(--mut)}
table{border-collapse:collapse;font-size:.85rem;margin:8px 0}th,td{border:1px solid var(--line);padding:3px 7px;text-align:left;vertical-align:top}
th{background:var(--card)}.scroll{overflow-x:auto}.note{background:var(--card);border-left:4px solid var(--mut);padding:8px 12px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media (max-width:800px){.pair{grid-template-columns:1fr}}
"""


def body():
    P = []
    a = P.append
    a('<h1>v11・v12 の図と数値（確認用）</h1>')
    a('<p>2026-10-21 作成。既にある図と数値を集めたもの（新しい学習・評価はしていない。B の表だけは事後の分析の結果）。'
      '詳しくは <code>docs/v12_experiment_log.md</code>（以下 ログ）、<code>docs/v12_summary.md</code>、<code>docs/v11_summary.md</code>。'
      '図はクリックで原寸。各図に「何を示すか・どう読むか・目視の確認」を 1 行ずつ付けた。</p>')
    a('<div class="note"><b>用語</b>：経験なし（random）＝ A-1 がランダムに歩いて学習したモデル、経験あり（mixed）＝ 半分は赤の方へ動いて学習したモデル。'
      'original ＝ 元の配置、swap_rg ＝ 赤と緑を入れ替えた配置。D6 ＝ 部品 G の代わりに A-2 の本当の快・不快をそのまま入れた参照の値。</div>')

    # 6 first: the table, so the reader has the verdicts up front
    a('<h2>まとめの表：1 段目の基準・関門と 2 段目の判定</h2>')
    a('''<div class="scroll"><table>
<tr><th>段</th><th>項目</th><th>v11（快・不快だけを入力、一から学習）</th><th>v12（運動を戻し、ときどき隠す。Encoder は v4 で凍結）</th></tr>
<tr><td rowspan="5">1 段目（自己）</td><td>価値と行動（M1(c)・M1(d)）</td><td><b>合格</b> 3/3（M1(d) 組の差 +0.0094）</td><td><b>合格</b>（G2、運動が隠れた一歩で 3/3、M1(d) 組の差 +0.011〜+0.012）</td></tr>
<tr><td>価値と色（S赤 − S緑）</td><td><b>合格</b> 3/3（+0.006〜+0.007）</td><td><b>合格</b>（G2 の中、+0.007〜+0.009）</td></tr>
<tr><td>自分側の積み上がり（入力を止める）</td><td><b>不合格</b> 0/3（3〜5 ステップで落ち着く）</td><td><b>合格</b>（G3 3/3、組の差 +0.068〜+0.075。注意：経験なしも 4 色の方へ同じくらい進む）</td></tr>
<tr><td>他者側の積み上がり（process-2 に A-2 の本当の快・不快）</td><td>陽性の形にならない（参考の検査）</td><td><b>合格</b>（G5a 3/3、G5b 2/3。注意：s1・s2 は CI の下限が 0 に近い、B1 の読み取りでは s0 だけ）</td></tr>
<tr><td>自他の書き方の揃い・Fig.5</td><td>揃い 0.16〜0.30、Fig.5 成り立たず</td><td>Encoder を v4 で凍結したので v4 と同じ（12 本とも一致、マージン +0.075）。<b>与えたもので、学んだものではない</b></td></tr>
<tr><td rowspan="5">2 段目（他者）</td><td>① T2（入れ替えの瞬間）</td><td><b>不合格</b>（s0・s2・s3 で 1/3）</td><td><b>不合格</b>（正式 1/3、4 組でも不合格）</td></tr>
<tr><td>② T5（好みだけが違う A-2）</td><td><b>合格</b> 2/3（ただし D6 の 1/10 以下、相関が支えない、組で向きがそろわない）</td><td><b>不合格</b>（正式 1/3）</td></tr>
<tr><td>部品 G が使った枠</td><td>Q2 の分散の 79〜94% が範囲 2〜5</td><td>71〜86% が範囲 2〜5（同じ形）</td></tr>
<tr><td>2 段目 b（探索、出力の形を限定）</td><td>—</td><td><b>不合格</b>（① 0/4、② 1/4）。4 組とも「残る抜け道」の形。案 A（参考、s0）も同じ形</td></tr>\n<tr><td>2 段目 c（探索、A-2 を隠す。主は隠した test）</td><td>—</td><td><b>不合格</b>（① 0/3、② 1/3）。「うまくいかない場合 a」。① は 3 組とも大きく逆向き</td></tr>
</table></div>''')

    a(fig(f'{V12}/report/figs/network_v11_v12.png', 'v11・v12 のネットワーク（清書）', '青が 1 段目で学習、橙が 2 段目で学習する部品 G、斜線が凍結した v4 の重み。赤字は共有モジュールへの入力。', True))
    a(fig(f'{V12}/report/figs/compare_v11_v12.png', 'v11 と v12 の比較', '(a)(b) 1 段目の価値と行動・色、(c) 自分側の積み上がり（v11 は不合格、v12 は合格）、(d) 他者側の積み上がり（v12 だけ）、(e) 2 段目の判定。', True))
    a('<h2>1. 予測画像の比較（同じエピソード・同じ時刻）</h2>')
    a('<p>上から：入力、次の視界（正解）、予測、予測の誤差、Fig.5 の再構成、A-2 の本当の視界、Fig.5 で取り出した A-2 の視点、その誤差。'
      't=1,5,10,20,50（t>0 は視覚を隠す）。v11 と v12 は同じ test のエピソード（共通の random test）。</p>')
    for c, lay, ep, chk in (('random', 'original', 0, True), ('mixed', 'swap_rg', 1, True),
                            ('random', 'swap_rg', 1, True), ('mixed', 'original', 0, True)):
        cj = '経験なし' if c == 'random' else '経験あり'
        lj = '元の配置' if lay == 'original' else '入れ替えた配置'
        a(f'<h3>{cj}・{lj}（エピソード {ep}、組 s0、ep200）</h3><div class="pair">')
        a(fig(f'{V11}/main/pred_images/pred_common_{c}_{lay}_ep{ep}_s0_e200.png', f'v11 {cj}',
              '予測の行は正解とほぼ同じ。Fig.5 で取り出した A-2 の視点（下から 2 行目）は A-2 の本当の視界に近くない' +
              ('（色の並びが合わず、緑の場所に赤が出る）。' if c == 'random' else '（大きな暗い塊が出る）。') +
              ('' if chk else '（この組み合わせは目視していないので、説明は他の例からの推定）'), chk))
        a(fig(f'{V12}/main/pred_images/pred_common_{c}_{lay}_ep{ep}_s0_e200.png', f'v12 {cj}',
              '予測は v11 と同じくらい。Fig.5 で取り出した A-2 の視点は、ランドマークの並びが A-2 の視界に近いが、赤・緑が黄色っぽく混ざる（Fig.5 の Decoder の描き方、ログ §22）。' +
              ('' if chk else '（この組み合わせは目視していないので、説明は他の例からの推定）'), chk))
        a('</div>')
    a(links([f'{d}/main/pred_images/pred_common_{c}_{lay}_ep{ep}_s{s}_e200.png' for d in (V11, V12) for s in (1, 2)
             for c, lay, ep in (('random', 'original', 0), ('mixed', 'swap_rg', 1))], '目視：済み（2026-10-22）'))
    a('<h3>2 段目：入れ替え（t=50）の前後の予測画像（T4 の例）</h3><div class="pair">')
    a(fig(f'{V12}/stage2/figs/t4_example_green_s0.png', '2 段目（正式）、組 s0',
          '上 2 行が正解（入れ替えなし／あり）、下 4 行が経験なし・ありの予測。黄色い線は予測画像から読んだ A-2 の方角。入れ替えのあと、予測の A-2 が新しい緑（左）の方へ向かう形は、経験ありでも見えない。下の分布は経験なしの方が平均が大きい（+0.82 対 +0.37）。', True))
    a(fig(f'{V12}/stage2b/figs/t4_example_green_s0.png', '2 段目 b、組 s0',
          '読み方は左と同じ。正式とほぼ同じ形（経験なし +0.69、経験あり +0.32）。', True))
    a('</div>')
    a(links([f'{V12}/stage2/figs/t4_example_{st}_s{s}.png' for st in ('green', 'pref') for s in range(4)] +
            [f'{V12}/stage2b/figs/t4_example_{st}_s{s}.png' for st in ('green', 'pref') for s in (0, 2)]))

    a('<h2>2. Fig.5（誤差の地図と、取り出した A-2 の視点）：v4・v11・v12</h2>')
    a('<p>各行の左 2 つの地図：A-2 の視点として取り出したもの（other_rec）・A-1 の再構成（self_rec）を、場のすべての位置の視界と比べた誤差（青が小さい）。'
      '○1 が A-1、□2 が A-2。other_rec の地図の青が □2 に来れば成り立つ。右は A-2 の本当の視界と取り出した視点。左半分が元の配置、右半分が入れ替えた配置。</p>')
    a(fig(f'{V11}/test_scripts/fig5_validation/fig5_maps_fig5_v4_exp1l1000_s0_ep100.png', 'v4（元論文 exp1 の再現、運動入力）',
          'other_rec の地図の青が □2 に来る。取り出した視点も、色を含めて A-2 の本当の視界に近い。', True))
    a(fig(f'{V11}/main/fig5/fig5_maps_v11_fig5_mixed_s0_s0_ep100.png', 'v11 経験あり s0',
          'other_rec の地図の青が □2 にも ○1 にも広がり、位置が定まらない（2 行目）。取り出した視点はぼやけ、暗い塊が出る。経験なしも同じ（ログ v11 §7.39）。', True))
    a(fig(f'{V12}/main/fig5/fig5_maps_v12_fig5_s0_s0_ep100.png', 'v12（Encoder は v4 で凍結）',
          'other_rec の地図の青が □2 に来て、v4 とほぼ同じ形。取り出した視点はランドマークの並びが合うが、赤・緑が黄色っぽい（Decoder の描き方）。v12 の Fig.5 は Encoder を凍結した結果で、学んだものではない。', True))
    a(links([f'{V11}/main/fig5/fig5_maps_v11_fig5_random_s0_s0_ep100.png', f'{V12}/main/fig5/fig5_perA2pos_v12_fig5_s0_s0_ep100.png',
             f'{V11}/main/fig5/fig5_perA2pos_v11_fig5_mixed_s0_s0_ep100.png']))

    a('<h2>3. 内部状態の PCA（論文 Fig.4c の作り）：v11・v12、経験なし・あり</h2>')
    a('<p>左から：h¹ を A-1 の位置で色づけ、h² を（h¹ の主成分に射影して）A-2 の位置で色づけ、h¹ を快・不快で色づけ、h¹ を A-1 から赤までの距離で色づけ、h² 自身の主成分（補足）。'
      '上の行が元の配置、下の行が入れ替えた配置。位置の色は角ごとに（+x,+y）黄、（−x,+y）青、（−x,−y）緑、（+x,−y）赤。</p>')
    for ver, d, note, note2 in (('v11', V11, 'h² を h¹ の主成分に射影すると 1 点に縮む（自他を別の書き方で表す）。',
                                 '経験ありは h¹ が快・不快と赤までの距離の軸に沿って細長く並ぶ。'),
                                ('v12', V12, 'h² を h¹ の主成分に射影しても A-2 の位置で広がる（自他が同じ書き方。Encoder の凍結による）。',
                                 '経験なしと同じく h² は A-2 の位置で広がる。h¹ の快・不快の傾きは両条件にある。')):
        a(f'<h3>{ver}</h3><div class="pair">')
        a(fig(f'{d}/main/pca/pca_random_s0_ep200.png', f'{ver} 経験なし s0',
              note + ' 快・不快の色の傾き（3 列目）は経験なしにもある（入力に由来）。', True))
        a(fig(f'{d}/main/pca/pca_mixed_s0_ep200.png', f'{ver} 経験あり s0',
              '読み方は左と同じ。' + note2 + ' PCA の違いは判定には使っていない（v11 ログ §7.26）。', True))
        a('</div>')
    a(links([f'{d}/main/pca/pca_{c}_s{s}_ep200.png' for d in (V11, V12) for c in ('random', 'mixed') for s in (1, 2)], '目視：済み（2026-10-22）'))

    a('<h2>4. 1 段目の主な結果</h2>')
    a(fig(f'{V11}/summary_figs/fig_m1d_colours.png', 'v11 M1(d) の色ごとの寄り',
          '棒が正なら、予測が「その色の方へ一歩動いた場合の視界」に寄っている。経験あり（赤系）だけ赤で大きく正、緑で小さく正。経験なし（灰色）は 4 色とも負。', True))
    a(fig(f'{V12}/main/s1eval/figs/accumulation_s1.png', 'v12 G3：入力を止めたあとの積み上がり',
          '左・中：入力（運動・視覚）を止め、快・不快を保ったまま、予測が何歩ぶん各色の方へ進んだか（d*）。経験ありは赤の方へだけ進み続け、ほかの 3 色はほぼ止まる。経験なしは 4 色とも同じように進む。右：傾き。v11 は両条件とも ≈ 0。', True))
    a('<div class="pair">')
    a(fig(f'{V12}/main/g5/s0/g5a_curves.png', 'v12 G5a：他者側の積み上がり（組 s0）',
          'process-2 に A-2 の本当の快・不快を止めて入れたとき、予測画像の A-2 が好きな色の方向（実線）・ほか 3 色（点線）へどれだけ進むか。経験あり（赤）は好きな色の方へ相対的に進み、経験なし（灰）は離れる。', True))
    a(fig(f'{V12}/main/s1eval/figs/m1d_colours_s1.png', 'v12 M1(d) の色ごとの寄り',
          'v11 の図と同じ読み方（運動が隠れた一歩）。赤に特有の寄りが経験ありで出る。', True))
    a('</div>')
    a(links([f'{V12}/main/g5/s{s}/g5a_curves.png' for s in (1, 2)] + [f'{V12}/main/g5/s{s}/g5b_bins.png' for s in (0, 1, 2)],
            '目視：済み（ログ §27・§30 の報告のとき）'))

    a('<h2>5. 2 段目</h2>')
    a(fig(f'{V12}/report/figs/q2_timeseries.png', '生成した Q2 と A-2 の本当の快・不快の時系列（組 s0、①緑好きの評価セット、3 エピソード）',
          '実線が Q2、点線が P_A2（範囲 0,1 が緑系、6,7 が橙系）、黒線は正式の範囲 2〜5 の平均。正式：Q2 は ±1 まで大きく振れ、P_A2 の形に沿わない。'
          '2 段目 b：範囲 0,1 は正、6,7 は負に、配置（エピソード 0 は入れ替え、1・2 は元）によらずほぼ固定される（符号 u が一定＝抜け道の形）。', True))
    a('<div class="pair">')
    a(fig(f'{V12}/stage2/figs/q2_vs_pa2_green_s0.png', '正式：Q2 と P_A2 の散布図・範囲 2〜5 の分布（①、s0）',
          '上：範囲 0,1,6,7 で Q2 と P_A2 の相関は弱く負（−0.16・−0.11）。下：P_A2 が常に 0 の範囲 2〜5 で Q2 は ±1 に張り付く（標準偏差 0.9）。', True))
    a(fig(f'{V12}/stage2b/figs/q2_vs_pa2_green_s0.png', '2 段目 b：同じ図（①、s0）',
          '範囲 2〜5 は作りの上で 0。範囲 0,1,6,7 の相関は ≈ 0（+0.05・−0.04）で、Q2 は ±1 か 0 に集まる。', True))
    a('</div>')
    a(fig(f'{V12}/stage2/figs/q2_vs_pa2_pref_s0.png', '正式：同じ図（②好みがエピソードごとに違うセットで学習、s0）',
          '①と同じ形。相関 −0.01・−0.11、範囲 2〜5 は ±1 に張り付く。', True))
    a(links([f'{V12}/stage2/figs/q2_vs_pa2_{st}_s{s}.png' for st in ('green', 'pref') for s in (1, 2, 3)] +
            [f'{V12}/stage2b/figs/q2_vs_pa2_{st}_s{s}.png' for st in ('green', 'pref') for s in (0, 2)]))
    a(fig(f'{V12}/report/figs/t2t5_ckpts.png', 'T2・T5 の G の時点ごとの値（全組、正式と 2 段目 b）',
          '赤の実線が経験あり、灰の破線が経験なし。仮説の向きは正。どの値も ±0.2 以内で、D6（T2 +0.60、T5 +0.42）よりずっと小さい。経験ありが経験なしの上にそろう形はない。', True))
    a(fig(f'{V12}/report/figs/s2_attempts.png', '2 段目の三つの試み（正式・2 段目 b・案 A・2 段目 c）の主の値',
          '黒い横線が経験ありの値、赤い点と縦線が組の差（経験あり − なし）と 95% CI、上の n/5 は合格の形の点の数（2 段目 c は隠した test）。合格の形の組も D6 よりずっと小さく、組・試みで向きがそろわない。2 段目 c の ① は 3 組とも大きく逆向き。', True))
    a(fig(f'{V12}/report/figs/t3_attempts.png', 'T3：process-2 を零・生成した Q2・D6・位置の代理にしたときの予測の誤差（正式・b・案 A・c）',
          '棒が平均、点が組 × 条件。どの試みでも生成が D6 よりずっと良い（部品は好みより予測に役立つ別の情報を流す）。(c) 2 段目 c の隠した test では、A-2 が隠れているフレームで生成と D6 の差が小さくなる。', True))
    a(fig(f'{V12}/report/figs/t4_dist.png', 'B4：T4 のエピソードごとの分布（①のモデル、ep200）',
          '正なら、入れ替えのあと予測の A-2 が新しい緑の方へ向く。箱は四分位、橙の線は中央値。どの組でも分布は広く、中央値は経験なしの方が大きいか同じ（経験ありで大きい形はない）。', True))

    a('<h2>B. 事後の分析（事前登録外。判定は変えない。ログ §41）</h2>')
    a('<p>2 段目で好みが「判定の測り方では見えないが、実はある」かを確かめる。B1：生成した Q2 から A-2 の好み（①配置、②エピソードの好み）を読む。'
      'B2：h²（他者側の内部状態）から読む。「位置だけ」は A-1・A-2 の位置の 2 次多項式だけの正答率で、「位置 ＋ 特徴」との差が、位置で説明できない上乗せ。'
      'B3：t=20 の Q2 を 15 ステップ止めたときに、予測画像の A-2 が好きな色の方へ進むか（G5a と同じ測り方）。</p>')
    a(table_b())

    a('<h2>C. 案：「A-2 を隠す 2 段目」（学習はまだ始めない。ログ §43）</h2>')
    a('''<ul>
<li><b>ねらい</b>：A-2 が見えている限り、部品 G は好みより位置を流す方が予測に役立つ（ログ §40）。A-2 を A-1 の視界から区間で隠し、「好みを知らないと A-2 の動きが予測できない場面」を作る。</li>
<li><b>実装</b>：A-2 を場の外に置いて描き直すと、A-2 だけが消えて他の画素は 1 つも変わらない（確認済み、1 枚 14 ms）。入力の視界だけを「A-2 なし」に替え、予測の正解は A-2 が写った元の視界のまま。読み込みの v12 版を新しいファイルで作る（既存のコードは変えない）。データ作り 約 30 分（CPU）、ディスク 約 2 GB。</li>
<li><b>隠す区間</b>：基準は 5〜15 ステップを交互（約 48%）。別案は 10〜30 ステップ（長いほど好みの利点は大きいが、学習が不安定になる心配）。</li>
<li><b>評価</b>：事前登録の test（見える）はそのまま主。隠した test を足し、隠れている区間の T5・Q2 と P_A2 の相関・B3 を副に。T2 は t=50 で見える／隠れているで分ける。</li>
<li><b>見込み</b>：層 1（抜け道）は出力の形（2 段目 b の案 B）で、層 3（位置が視覚から取れる）は隠すことでふさげる。層 2（好みを伝えても改善が小さい）は残るが、隠れている間は寄りが積み上がるので利点は見える場合より大きい。うまくいかない場合、G は最後に見えた位置と動きを範囲ごとの強さで流し、u は一定のまま、と予想する。</li>
<li><b>時間</b>：学習 約 8.5 時間（1 GPU に 2 本、3 組なら GPU3〜7）、評価 各組 約 1 時間。</li>
</ul>''')
    return '\n'.join(P)


def main():
    os.makedirs(OUT, exist_ok=True)
    page = ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>v11・v12 確認資料</title><style>{CSS}</style></head><body>{body()}</body></html>')
    open(os.path.join(OUT, 'index.html'), 'w').write(page)
    print('saved', os.path.join(OUT, 'index.html'))


if __name__ == '__main__':
    main()
