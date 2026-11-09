# 追加実験（作業記録 §35〜§39）の結果の一覧と、記録の残り方

- 作成：2026-10-31（第13弾の後）。既存のファイルは変えず、確認と一覧づくりだけをした。
- 追記：2026-11-01（第14弾の段階Cの行と報告のファイルを足した）。
- 追記：2026-11-01（第15弾 D1・D2 の行と報告のファイルを足した）。2026-11-02（第16弾の行と報告のファイルを足した）。2026-11-02（第17弾の行と報告のファイルを足した）。2026-11-03（第18弾の行と報告のファイルを足した）。2026-11-05（第19弾の行と報告のファイルを足した）。2026-11-06（第20弾の行と報告のファイルを足した。同じ日に第20弾の追加 $I$ の行、第21弾・第22弾・第23弾の行と報告のファイルを足した）。 2026-11-07（第24弾の行と報告のファイルを足した）。
- 作業場所：`.worktrees/v11/my_research/preference_inference`。枝 `feature/v11-self-experience`。この一覧の作成の時点で、手元の枝は origin と同じ（push 済み）。
- 「事前登録」は計算の前にコミットした判定の形。「事前登録外」は、手順だけを計算の前にコミットした記述・探索。

## 1. 結果の一覧

| 節 | 何を | 扱い（事前登録のコミット） | 結果のコミット | 結果 |
|---|---|---|---|---|
| §35.1〜35.3・35.7 | **E2**：正しい好み（P）と反対の好み（−P）を入れたときの他者の予測の誤差の差 $U_{\mathrm{pref}}$。主：経験50% − 経験なし（好み混合・FP・1〜50） | 事前登録 `cdd5043` | `1380e10` | **主の判定は不成立**：+0.0010 [−0.0050, +0.0071]。副：経験80% − 経験なし +0.0139 [+0.0016, +0.0262] など一部だけ下限 > 0 |
| §35.4〜35.6・35.11 | **E1**：経験50% の符号無作為条件（6 組）の学習・評価。主：$d_P$ の 符号無作為（経験50%）− 経験50% | 事前登録 `a1770ba`（学習の前） | `b5c9b09` | **主の判定は成立**：−0.0707 [−0.0787, −0.0627]、6/6。副（符号に従う分 ①②、E2 の $U_{\mathrm{pref}}$）はどれも不成立 |
| §35.9・37.7 | **E1 の副**：位置のずれの $U^{\mathrm{pos}}_{\mathrm{pref}}$ の 経験50% − 符号無作為（経験50%） | 事前登録 `9e86dd4`（E1 の評価の前） | `bd7cf3f` | **不成立**：−0.124 [−1.421, +1.172]、正 2/6 |
| §35.8・35.10 | E2 の追加：位置のずれ $U^{\mathrm{pos}}$ と「赤の形」への偏り（記述） | 事前登録外（手順 `aa7e629`） | `b5c9b09` | 経験50% − 経験なしの $U^{\mathrm{pos}}_{\mathrm{pref}}$ は 0 を含む。経験80% の赤の形への偏り +0.017 [+0.005, +0.028] |
| §35.12〜35.13 | 論文の芯の見直しのための回収と、E2 のエピソードごと・フレームごとの分析（記述） | 事前登録外（手順 `e8733b5`） | `92040f1` | 二択の BA はどの条件も 0.5 前後。経験80% の $U_{\mathrm{pref}}$ はフレーム 1〜20 で正、41 以降で負 |
| §36.1〜36.6 | **V-dir 段階A**：保存配列での $Q_e$ の回帰（好み・向き・赤の側） | 探索的・事前登録外（手順 `fa20545`） | `d34ea42` | 経験50%・80% で $b_{\mathrm{dir}} > 0$。oracle・FP の $b_{\mathrm{dir}}$ は初期位置で大半が消える |
| §36.7〜36.8 | 段階Aの追加：$c^{\pm}$ 経験50% の $b_{\mathrm{pref}}$、初期位置ありの形、必要なエピソード数 | 探索的（手順 `5781907`） | `4cea461` | $c^{\pm}$・FP の $b_{\mathrm{dir}}$ は初期位置を入れても残る。$N_{80}$ の目安 |
| §37.1〜37.2 | 段階B：データを作る前の確認 | 手順 `a73e74a` | `3e26c84` | 層ごとの向きの偏りで、止まる条件に当たった（12 セル中 9） |
| §37.3〜37.4 | 改訂2 §1.6：符号無作為（経験50%）の $c^{\pm}$（段階Aのデータ）、P3 の $N_{80}$、1200 での見込み | 探索的（手順 `afba126`） | `0b8f4b0` | P3 の $N_{80}$ 53。食い違うエピソードの見込み 311。止まる条件に当たらない |
| §37.5〜37.6・37.8 | **V-dir 段階B**：新しいデータ（1200）での確認的な検証。族 P（$b_{\mathrm{pref}}$）・族 D（$b_{\mathrm{dir}}$） | **事前登録 `4216497`**（データを作る前）。データの記述 `bd7cf3f`、集計のコード `b500bac`（出力を読む前） | `vdir_stageB.md` のコミット（下の §2） | **族 P 成立**（P1 +0.0378、P2 +0.0388、P3 +0.0400。t・2段とも下限 > 0）。**族 D 成立**。食い違うエピソードでの経験50% の好みの正解率は 0.360（0.5 未満）。感度 (v) で $b_{\mathrm{pref}}$ は +0.008 に縮む（正のまま） |
| §38 | **段階C**：G の最小限の診断。主（L1）：$\tilde q_e$（$Q2$ の $c^{+}$ への射影）の回帰の $b^{q}_{\mathrm{pref}}$。副（L2）：$Q2$ からの好みの読み出し。記述（L3）：(O)(S)(H)(H0) の読み出し、一般化、勾配 | **事前登録 `3ade5ed`**（G の評価の前）。修正 `1de4a37`（健全性確認の不合格の後、実データの集計の前。読み出し器の C を入れ子の交差検証で選ぶ）。集計のコード `c1a0a17` | `stageC_G.md` のコミット | **第1段は不成立**（Q1a −0.037、Q1b −0.094。Q1c・第2段は判定せず）。(L2) は経験50% の BA 0.816・$\Delta\mathrm{LL}$ +0.029 で成立、経験50% − 経験なしは不成立（$\Delta\mathrm{LL}$ −0.021）。別紙のパターン B に最も近い。(H0) に説明のつかない点（未確認） |
| §39 | **第15弾 D1・D2**：行動の手がかりだけの基準（B1 要約の手がかり・B2 軌跡）からの好みの読み出しと、実験2の $Q_e$・実験3の $q_e$・$Q2$ の読み出しのエピソードごとの相関 | 事前登録外の記述（手順 `dea5dd3`） | `round15_D1D2.md` のコミット | B1 BA 0.833・B2 BA 0.858。読み出しの BA は 10 中 8 で B1・B2 と CI が重なる。一般化は B1 0.000・B2 0.086 で、どの読み出しより下。$Q_e$ と $Q2$ の読み出しの確率の相関は経験50% +0.626 |
| §39.3〜39.4 | **第15弾 b1・c1**：好みの形の差（$U_{\mathrm{pref}}$）と $G$ が下げた他者特徴の誤差の比、正しい好みの役立ち方（$U_{\mathrm{use}}$・$U^{\mathrm{pos}}_{\mathrm{use}}$）の条件の間の差 | 事前登録外の記述（手順 `9e90b64`） | `round15_b1c1.md` のコミット | 比は経験50% 0.014・経験なし 0.007。$U_{\mathrm{use}}$ の経験50% − 経験なし +0.0264（下限 > 0）、− 符号無作為 −0.0058（0 を含む）。$U^{\mathrm{pos}}_{\mathrm{use}}$ +2.14（下限 > 0）／+1.14（0 を含む） |
| §40 | **第16弾**：生成器の損失と位置の寄与の診断（$Q2$ を差し替えた順伝播。$D_G$、$R_1(s)$、$R_2(s)$、$S_{\mathrm{pos/res/ax/orth}}$）と小さい計算 E | 探索・事前登録外（手順 `8fcb54f`） | `round16_results.md` のコミット | 経験50%：$R_1/D_G$ は −0.087〜+0.065（t の CI はどれも 0 を含む）、$R_2(1)$ +0.033（下限 > 0）、$S_{\mathrm{pos}}$ 0.48・$S_{\mathrm{res}}$ 0.48・$S_{\mathrm{ax}}$ 0.14。読み方の表の行3（$s=1$）・行4 |
| §41 | **第17弾**：生成器の出力が運ぶ行動の情報（$Q2$ を行動の量 F0〜F5 で交差当てはめし、当てはめ値と残りを第16弾の経路に入れた。$R^2_F$、$S_F$、$S^{\mathrm{res}}_F$、$\Delta_F$、$S_{F5}-S_{F4}$） | 探索・事前登録外（手順 `a25bc4e`） | `round17_results.md` のコミット | 経験50%：$S_{F4}$ 0.815・$S^{\mathrm{res}}_{F4}$ 0.341・$R^2_{F4}$ 0.634、未来の変位の増分 +0.002、$S_{F5}-S_{F4}$ +0.000 [−0.001, +0.001]。読み方の表はどの行にもそのままは当たらない（最も近いのは行1。$S^{\mathrm{res}}_{F4}$ が小さくない） |
| §42 | **第18弾**：実験2を A-2 の好み別に分ける（段階Bの主の回帰の $b_{\mathrm{red}}$、好み別の $b_{\mathrm{pref}}$（向きの調整あり・なし）、好み × 配置のセル、食い違う場面、$\pm P^{\mathrm{A2}}$、E2 の好み別の CI） | 探索・事前登録外（手順 `ae13017`） | `round18_results.md` のコミット | 経験50%（c±）：$b_{\mathrm{red}}$ −0.0028（0 を含む）。好み別の $b_{\mathrm{pref}}$ は赤好き +0.0327・緑好き +0.0420（両方の CI の下限 > 0）。読み方の表の行1。他者自身の視界から作る候補には赤への偏りの形 |
| §43 | **第19弾**：今後の研究の候補の下調べ（§1 作れるかの確認 5 点、§2.1 観察が弱い場面での赤への寄り、§2.2 Encoder-2 の出力からの他者の視点の赤・緑の方角） | §1 は読むだけ。§2 は探索・事前登録外（手順 `c1dd36b`） | `round19_results.md` のコミット | §1：5 点とも手を入れれば作れる。§2.1：経験50% の赤への偏りは 0.51〜0.54 で CI が 0.5 を含む（寄らない）。§2.2：赤・緑の方角 $R^2$ 0.61〜0.90（位置だけ 0.13〜0.23） |
| §44 | **第20弾**：途中で赤と緑を入れ替える検査（Woodward 型。新しい評価データ 600 × N・O・S・R × `p_rule` 0.8・0.5、学習なし。$D_{\mathrm{swap}}$・$D_{\mathrm{noswap}}$、視覚 V0・V1・V2 × 入力 G・T・Z、自己の誤差・追跡の乱れ） | 探索・事前登録外（データの手順 `99d3bc7`、§2 の手順 `2d79c1c`） | `round20_results.md` のコミット | 主（経験50%・G・V1・0.8）の $D_{\mathrm{swap}}$ −0.0615 [−0.0777, −0.0453] [−0.0707, −0.0519]（4/4 が負）、$D_{\mathrm{noswap}}$ +0.0575。T も負（−0.0394）。読み方の表の行1（場所に結びつけた予測）。追跡の乱れなし |
| §44.5〜44.6 | **第20弾の追加**：色の配置の効果 $I = D_{\mathrm{swap}}+D_{\mathrm{noswap}}$（保存配列だけ。G・T・Z × V1・V2・V0 × 0.8・0.5、条件の差・T − Z・G − Z） | 探索・事前登録外（手順 `7c04484`） | $I$ の結果のコミット | 入力 T で $I>0$ の行はない（経験50%・V1・0.8 は −0.0173、≈0）。主は −0.0041（≈0）。V1 はほぼ全部 ≈0。色の配置を使っていない読み |
| §45 | **第21弾**：自己の経路で、入れ替えの後に「赤の新しい場所へ進む」と予測するか（新しい評価データ 600 × swap・noswap、A-2 はとどまる、学習なし。$C=\Delta_{\text{入れ替えなし}}-\Delta_{\text{入れ替えあり}}$、自己 K = 15・他者 K = 10、保つ・反転・零 × V1・V0） | 探索・事前登録外（データの手順 `7cc6885`、§2 の手順 `83a6916`） | `round21_results.md` のコミット | 主（経験50%・V1・保つ）の $C_{\mathrm{self}}$ +0.2494 [+0.2149, +0.2839] [+0.2256, +0.2753]（6/6 が正）、$C_{\mathrm{other}}$ +0.0247（≈0）。読み方の表の行2（自己から他者への流用の問題）。$C_{\mathrm{self}}$ の経験50% − 経験なし・− 符号無作為はどちらも $>0$（自分の価値の経験に依存） |
| §46 | **第22弾**：他者の経路は快・不快の向きを使うか（第21弾のデータと流し方、学習なし。経験80%・符号無作為（経験80%）、他者の経路への入力 X1（2倍・4倍）・X2（自己とそろえる）・X3（1 つの範囲に +0.28）、読み出し img（予測画像の A-2 の位置）と feat（予測した他者の視点の特徴の照合）） | 探索・事前登録外（手順 `732d4a7`） | `round22_results.md` のコミット | 主（img）はどの行にも当たらない（他者の保つ − 反転 ≈0、$C_{\mathrm{other}}$ は ≈0 か CI が食い違う）。補足（feat）は経験50%・経験80% とも行1（経験50% の $C_{\mathrm{other}}$ +0.1871、6/6 が正）。img と feat は食い違う。他者の快・不快は自己より小さくない（X2 は 0.85 倍の縮小） |
| §47 | **第23弾**：特徴の照合は「動き」を読んでいるか（2.1 陽性対照：第20弾のデータ `p_rule` 0.5 で N/O・S/R の正答率、本当の特徴・本当の画像・予測した特徴・予測画像。2.2 見え方と動き：第21弾のデータで範囲 E・B・R に +0.28、$M_{\mathrm{feat}}$・$M_{\mathrm{img}}$。学習なし） | 探索・事前登録外（手順 `419d0e6`） | `round23_results.md` のコミット | 2.1 はどの行にも当たらない（本当の特徴の BA 0.774、予測した特徴 0.72〜0.75、予測画像 0.80〜0.84）。2.2 もどの行にも当たらない（$M_{\mathrm{feat}}$ E $>0$・B ≈0・R $>0$、経験の差は R だけ）。事後の記述：範囲 2〜5 には学習の間に快・不快が入らない |
| §48 | **第24弾**：「自分を出発点にして、他者を見て直す」学ばせ方は作れるか（下調べ、学習なし。入れ替えを入れた段階IIのデータの作り方、手がかりの数、学ばせる部分の分け方・恒等写像の一致・勾配、損失、評価、時間の見積もり、§2 出発点の $C_{SO}$） | 探索・事前登録外（下調べ） | `round24_results.md` のコミット | §1 はどれも作れる（§1.5 は条件つき：生成器は 3 条件 × 組1〜4 だけ）。手がかりは 1 エピソード 18〜28。恒等写像で始めるとビット単位で一致。§2 の生成器の出力を入れた $C_{SO}$ は 3 条件とも ≈0（経験50% +0.0238） |
| §49 | **第25弾**：「自分を出発点にして、他者を見て直す」学ばせ方の試し（組1。先生の了承の前。新しいデータ `paper_r25_swap50`（入れ替え 50%、`p_rule` 0.5）、足した層 8→8 + 生成器の段1（M・N、100 エポック）、共有の LSTM も学ぶ段2（1/10・1/100、50 エポック）、5 エポックごとの $D$ と自己の見張り） | 探索・事前登録外（§0 の一覧 `07bc6b8`、§2 の手順 `dd1a254`） | `round25_results.md` のコミット | §2.3 の止まる条件に 2 つ当たる：段2 の学習率（1/10・1/100 とも $C_{\mathrm{self}}$ が半分を割る条件がある）、範囲 2〜5（M では $D$ が動かず N でだけ動く）。§2.2 には記述として行4（$D_{\mathrm{end2}}\approx0$）に近い。全組には進まない。既存のものの sha256 は 9341／9341 が同じ |
| §50 | **第26弾**：なぜ身につかないか（§1 学習の信号 $S_{\mathrm{WT}}=L(W)-L(T)$ を段階Iの予測器 4 条件 × 組1〜6・V／B で、学習なし。§2 足した層だけを学ばせ、生成器の代わりに T を入れる入口の上限（3 条件 × V／B × 組1〜3、50 エポック）。§3.1 A-1 の動きと快・不快の結びつき（3 つのデータ）。§3.2 段階Iのデータを半分混ぜた段2（1/10・1/100）） | 探索・事前登録外（手順と計算の前の一覧 `bfcbbc5`） | `round26_results.md` のコミット | §1 行2（主の $S_{\mathrm{WT}}$ V +0.229・B +1.403、>0）。§2 記述として行3（経験50% は 3 組とも V・B で動かない）。§3 行1（段階IIのデータの A-1 は無作為、混ぜると $C_{\mathrm{self}}$ は 1.11 倍以上を保つ。$D$ は ≈0）。全組には進まない。既存のものの sha256 は 10015／10015 が同じ |
| §51 | **第27弾**：正しい好みの得はどこから来るか、最も緩い上限で検査の量は動くか（§1 第26弾 §1 の $S_{\mathrm{WT}}$ を A-2 の好み × 成分（A-1 の視界の L1 を A-2 の周りの領域とそれ以外に分ける、他者の特徴、自己の特徴）に分けた、学習なし、組1〜6。§2 生成器の代わりに T、流し方 B、足した層 + 共有の LSTM（1/10）、その条件の段階Iのデータを半分混ぜる、3 条件 × 組1〜3、50 エポック） | 探索・事前登録外（手順と計算の前の一覧 `b53711a`） | `round27_results.md` のコミット | §1 好み別は行2（赤好き +1.350・緑好き +1.457、>0）、成分別は行3（A-2 の周りの L1 +0.0046、≈0。L1 全体は CI が食い違う）。§2 行3（$D_{T,\mathrm{late}}$ 経験50% ≈0、符号無作為 >0、経験なしは食い違う。区切りの規則の文言には当たらず、論文の会話に返す）。全組には進まない。既存のものの sha256 は 10565／10565 が同じ |
| §52 | **第28弾**：自分の価値の経験は、他者の動きの予測を学ぶのを妨げるか（§1 第27弾 §2 の学び方（T、流し方 B、足した層 + 共有の LSTM（1/10）、段階Iのデータを半分混ぜる）を組4〜6 にも行い、3 条件 × 6 組にまとめた。手順1 は第26弾 §2 の B を組4〜6 で。§2 経験50% で段階Iのデータを混ぜない形、組1〜6、50 エポック） | 探索・事前登録外（手順と計算の前の一覧 `5f17e1f`） | `round28_results.md` のコミット | §1 行3（6 組で 経験50% − 経験なし −0.136・経験50% − 符号無作為 −0.072、どちらも ≈0。第27弾の組1〜3 の形は続かない）。§2 行3（50 エポック目の $C_{\mathrm{self}}$ の対の差 −0.009 ≈0。混ぜなくても下がらず、競合は確かめられない。$D_{T,\mathrm{late}}$ の対の差 +0.157 ≈0）。方向2 には進まない。既存のものの sha256 は 10850／10850 が同じ |
| §53 | **第29弾**：新しい仕組みを作る前の確かめ（§0 動きの表し方・A-2 の規則・段階Iの過程2の入力・元論文の Motion Generator の入口。§1 過程2の運動の欄に動きを入れたときの予測画像と他者の視点の特徴、4 条件 × 組1〜6 × 8 方向、学習なし。§2 快・不快 → 次の自分の動きの小さな学習。§3 自分の経験だけで学んだ Encoder-1 の読み出しを他者の視点に当てる。§4 理想の観察者が好みを見分ける歩数、`p_rule` 0.5・0.8） | 探索・事前登録外（手順と計算の前の一覧 `ba82f9b`） | `round29_results.md` のコミット | §1 行1（$M_{\mathrm{img}}$ +0.250 >0、量は入れたとおりの約 1/4）。§2 行1（0.471、最良 0.496）。§3 行3（実際の他者の視点 0.442、予測した他者の視点 −0.113）。§4 行1（4 歩で 95%）。既存のものの sha256 は 11487／11487 が同じ |
| §54 | **第30弾**：他者の快・不快を画像の空間で作れるか（§0 復号器・A-2 の視界の画像・快・不快の計算・描画の設定。§1 自分の視界の画像 → 赤 − 緑の小さな網を他者の実際の視界に当てる、種 1〜6。§2 予測した他者の視点の特徴を復号して網を当てる（既存の Visual Decoder が主、新しい復号器は記述）、観察中・入れ替えの後に止めた予測、3 条件 × 組1〜6。§3 予備：画像から A-2 の位置と配置を読み、幾何で作る） | 探索・事前登録外（手順と計算の前の一覧 `9fd388b`） | `round30_results.md` のコミット | §1 行1（0.986）。§2 主は行4（(a) −0.02、赤と緑の区別が消える）、新しい復号器は行3 の形（0.85／0.60／0.28）。§3 観察中は行1（0.80）、止めた予測は行2（0.66）。既存のものの sha256 は 11684／11684 が同じ |
| §55 | **第31弾**：元論文と同じ形で、他者の視界の復号を確かめ直す（§1 Encoder-2 の入力・第29弾 §3 と第30弾 §2 で入れたもの・既存の復号器の学び方・元論文の Fig.5e の値。§2 格子の全配置（[−9, 9]²、重なりを除く、2 配置、259,920）で A-1 の視界 → Encoder-2 → 既存の復号器、3 つの画素の差・位置の量・色の量、§2.4 Encoder-2 → A-2 の赤 − 緑の診断のリッジ） | 探索・事前登録外（手順と計算の前の一覧 `ca8a920`） | `round31_results.md` のコミット | $d(\hat v^2, v^2)$ 0.105 < $d(\hat v^2, v^1)$ 0.157（元論文の向き）。位置の量 0.655、色の量 −0.066、Encoder-2 の色の読み出し 0.700（境目）。決め方のとおりでは行4（位置の量が目安の下）、形は行2 に近い。第30弾 (a) はすでに元論文と同じ入れ方だった。既存のものの sha256 は 14432／14432 が同じ |

## 2. 記録のファイル（すべて git に入っていて、push 済み）

| 種類 | ファイル | 最後のコミット |
|---|---|---|
| 作業記録（事前登録・手順・結果・計算後の変更の履歴を含む） | `docs/paper_extra_log.md` §35〜§37 | この一覧と同じコミット（§37.9） |
| 結果の要約 | `docs/paper_extra_results_log.md` | この一覧と同じコミット |
| E2・E1 の報告 | `paper/check/e2_e1_results.md` | `92040f1` |
| 段階Aの報告 | `paper/check/vdir_stageA.md` | `4cea461` |
| 段階Bの報告 | `paper/check/vdir_stageB.md` | `af2c17e` |
| 段階Cの報告 | `paper/check/stageC_G.md` | 段階Cの行を足したコミット |
| 第15弾 D1・D2 の報告 | `paper/check/round15_D1D2.md` | 第15弾の行を足したコミット |
| 第15弾 b1・c1 の報告 | `paper/check/round15_b1c1.md` | b1・c1 の行を足したコミット |
| 第16弾の報告 | `paper/check/round16_results.md` | 第16弾の行を足したコミット |
| 第17弾の報告 | `paper/check/round17_results.md` | 第17弾の行を足したコミット |
| 第18弾の報告 | `paper/check/round18_results.md` | 第18弾の行を足したコミット |
| 第19弾の報告 | `paper/check/round19_results.md` | 第19弾の行を足したコミット |
| 第20弾の報告 | `paper/check/round20_results.md` | 第20弾の行を足したコミット |
| 第21弾の報告 | `paper/check/round21_results.md` | 第21弾の行を足したコミット |
| 第22弾の報告 | `paper/check/round22_results.md` | 第22弾の行を足したコミット |
| 第23弾の報告 | `paper/check/round23_results.md` | 第23弾の行を足したコミット |
| 第24弾の報告 | `paper/check/round24_results.md` | 第24弾の行を足したコミット |
| 第25弾の報告・既存のものの一覧 | `paper/check/round25_results.md`、`round25_existing_inventory.md`・`round25_existing_inventory_sha256.txt`・`round25_existing_inventory_after.md` | 第25弾の行を足したコミット（一覧の前は `07bc6b8`） |
| 第26弾の報告・既存のものの一覧 | `paper/check/round26_results.md`、`round26_existing_inventory.md`・`round26_existing_inventory_sha256.txt`・`round26_existing_inventory_after.md` | 第26弾の行を足したコミット（一覧の前は `bfcbbc5`） |
| 第27弾の報告・既存のものの一覧 | `paper/check/round27_results.md`、`round27_existing_inventory.md`・`round27_existing_inventory_sha256.txt`・`round27_existing_inventory_after.md` | 第27弾の行を足したコミット（一覧の前は `b53711a`） |
| 第28弾の報告・既存のものの一覧 | `paper/check/round28_results.md`、`round28_existing_inventory.md`・`round28_existing_inventory_sha256.txt`・`round28_existing_inventory_after.md` | 第28弾の行を足したコミット（一覧の前は `5f17e1f`） |
| 第29弾の報告・既存のものの一覧 | `paper/check/round29_results.md`、`round29_existing_inventory.md`・`round29_existing_inventory_sha256.txt`・`round29_existing_inventory_after.md` | 第29弾の行を足したコミット（一覧の前は `ba82f9b`） |
| 第30弾の報告・既存のものの一覧 | `paper/check/round30_results.md`、`round30_existing_inventory.md`・`round30_existing_inventory_sha256.txt`・`round30_existing_inventory_after.md` | 第30弾の行を足したコミット（一覧の前は `9fd388b`） |
| 第31弾の報告・既存のものの一覧 | `paper/check/round31_results.md`、`round31_existing_inventory.md`・`round31_existing_inventory_sha256.txt`・`round31_existing_inventory_after.md` | 第31弾の行を足したコミット（一覧の前は `ca8a920`） |
| この一覧 | `paper/check/extra_results_index.md` | この一覧と同じコミット |
| コード（計算・集計・図・実行） | `analyze/paper_extra/e1_*.py`・`e2_*.py`・`e2pos_*.py`・`c_run.py`・`train_paper_e1.py`・`run_e1_*.sh`・`run_e2*.sh`・`vdir_*.py`・`vdirb_*.py`・`run_vdir_c.sh`・`run_vdirb_eval.sh`、`simulation/make_signrand_paper_e1.py`・`collect_data_vdir_b.py` | 各コミット（`git log -- <file>`） |
| 設定 | `config/exp/paper_extra/E1_signrand50/*.yml`（7 ファイル） | `a1770ba` |

- **git に入っていないもの（慣例で、入れていない）**：
  - 指示書：`paper/check/paper_instructions_round11.md`、`docs/paper_instructions_round12.md`・`round13.md`・`round13_rev2.md`。
    - 中身の要点は作業記録の各節に写してある。
  - 図：PDF・PNG（下の §3）。
  - 例外：第20弾の指示書 `paper/check/paper_instructions_round20.md` は、指示書の指定でコミットした（`99d3bc7`）。
  - 例外：第21弾の指示書 `paper/check/paper_instructions_round21.md` は、§1 のコミット（`7cc6885`）に入れた。
  - 例外：第22弾の指示書 `paper/check/paper_instructions_round22.md` は、§2 の手順のコミット（`732d4a7`）に入れた。
  - 例外：第23弾の指示書 `paper/check/paper_instructions_round23.md` は、指示書の指定でコミットした（`419d0e6`）。
  - 例外：第24弾の指示書 `paper/check/paper_instrctions_round24.md`（ファイル名は原本のまま）は、結果のコミットに入れた。

## 3. git に入っていない配列・データ・チェックポイント（2026-10-31 14:45 の大きさ）

`data/result/paper_extra/` は無視の設定の対象ではないが、慣例でコミットしていない（git に入っているファイルは 0）。`data/data/` は `.gitignore` の対象。

| 場所 | 中身 | 大きさ | 作り直す手順（作業記録の節） |
|---|---|---|---|
| `data/result/paper_extra/E1_signrand50/s1_signrand50_mse/<s>/model/` | E1 の MSE 期間のチェックポイント（6 組 × 201） | 4.6 GB | §35.4・35.6：`run_e1_train.sh <seed> <gpu>`（`train_paper_e1.py`、設定 `config/exp/paper_extra/E1_signrand50/`、`--cudnn_deterministic`）。GPU での学習がビット単位で再現するかは確かめていない（不明） |
| `data/result/paper_extra/E1_signrand50/s1_signrand50_l1_s<s>/` | E1 の L1 期間のチェックポイント | 6 × 62 MB | 同上（連鎖の後半） |
| `data/data/paper_s1_mixed50_signrand/signs.h5`・`signs.json` | E1 の学習データの符号の反転の表（データ本体は `v11_s1_mixed` を読むだけ） | 32 KB | §35.4：`simulation/make_signrand_paper_e1.py`。鍵 `paper_c_signrand/a1_sign`（経験80% の符号無作為と同じ） |
| `data/result/paper_extra/E1_signrand50/eval/` | E1 の評価（G1〜G5、$d_P$、符号に従う分など） | 4.4 MB | §35.4：`run_e1_eval.sh <seed>` |
| `data/result/paper_extra/E2/runs/` | E2 のエピソード × フレームの誤差（true・opp・零） | 439 MB | §35.2・35.6：`run_e2.sh`（`e2_run.py`）。2 回目の一致で再現を確かめた |
| `data/result/paper_extra/E2/pos/` | 位置のずれ（§35.8・35.9）の配列と集計 | 394 MB | §35.8：`run_e2pos.sh`、§35.9：`run_e2pos_sr50.sh`。集計は `e2pos_aggregate.py` |
| `data/result/paper_extra/E2/`（`summary.json`・`tables.md`・図）、`E2/episode/` | E2 の集計と図、§35.13 の集計 | 1 MB 未満 | `e2_aggregate.py`・`e2_fig.py`・`e2_episode.py`・`e2_episode_fig.py` |
| `data/result/paper_extra/Vdir_A/` | 段階Aの集計・点ごとの係数・表・図 | 32 MB | §36：`vdir_stageA.py`・`_tables.py`・`_fig.py`・`_add.py`・`_add_tables.py`（保存配列だけから。CPU） |
| `data/result/paper_extra/Vdir_B/c_sr50/` | 符号無作為（経験50%）の $c^{\pm}$（段階Aのデータ） | 約 90 MB | §37.3：`run_vdir_c.sh`（GPU）。2 回目がビット単位で一致 |
| `data/result/paper_extra/Vdir_B/check/` | §1 の確認の集計 | 1 MB 未満 | §37.1・37.3：`vdir_stageB_check.py`・`_check2.py` |
| `data/data/paper_vdir_b_pref/data.h5` | 段階Bの評価データ（1200） | 749 MB | §37.5.2・37.6：`xvfb-run --auto-servernum python simulation/collect_data_vdir_b.py`、鍵 `paper_vdir_stageB_pref_20261031`、セルの種 4293076338、sha256 の先頭 `1c3e8bff6e653cff`。既存のデータの作り直しでビット単位の一致を確かめた |
| `data/result/paper_extra/Vdir_B/eval/inverse/` | 段階Bの評価（331 ファイル、エピソード × フレーム） | 約 1.8 GB | §37.5.3：`run_vdirb_eval.sh <条件…>`（`vdirb_eval.py`）。2 回目がビット単位で一致 |
| `data/result/paper_extra/Vdir_B/data/`・`analysis/` | 段階Bのデータの記述、集計・表・図 | 数 MB | `vdirb_data.py`、`vdirb_analysis.py`・`vdirb_tables.py`・`vdirb_fig.py` |
| `data/data/paper_r20_swap_p80/`・`paper_r20_swap_p50/` | 第20弾の評価データ（各 `data_{N,O,S,R}.h5`、600 エピソード、81 フレーム） | 82 MB・81 MB | §44.1：`xvfb-run --auto-servernum python simulation/collect_data_round20.py --p_rule <0.8・0.5> --variant <N・O・S・R>`、鍵 `paper_round20_swap_20261106` |
| `data/result/paper_extra/Round20/` | 第20弾の確認（`check/`、約 90 MB）、評価（`eval/`、228 ファイル、約 2.1 GB）、集計・表・図（`analysis/`） | 約 2.2 GB | §44.2〜44.4：`round20_data_check.py`、`run_round20_eval.sh <w> 9`（`round20_eval.py`）、`round20_aggregate.py` |
| `data/data/paper_r21_swap/` | 第21弾の評価データ（`data_{swap,noswap}.h5`、600 エピソード、41 フレーム） | 17 MB | §45.1：`xvfb-run --auto-servernum python simulation/collect_data_round21.py`、鍵 `paper_round21_swap_20261106` |
| `data/result/paper_extra/Round21/` | 第21弾の照合する視界（`views/`、16 MB）、確認（`check/`、25 MB）、評価（`eval/`、120 ファイル、74 MB）、集計・表・図（`analysis/`）、試し（`smoke/`・`synthetic/`、89 MB） | 約 225 MB | §45.1〜45.3：`round21_views.py`、`round21_data_check.py`、`round21_eval.py check`、`run_round21_eval.sh <w> 9`（`round21_eval.py main`）、`round21_aggregate.py` |
| `data/result/paper_extra/Round22/` | 第22弾の確認（`check/`）、照合する視界（`views/`、28 MB）、評価（`eval/` 150 ファイル 104 MB、`eval21/` 30 ファイル 19 MB）、集計・表・図（`analysis/`）、試し（`synthetic/`、141 MB） | 約 290 MB | §46.1〜46.3：`round22_check.py`、`round22_views.py`、`round22_eval.py check`、`run_round22_eval.sh <w> 9`（`round22_eval.py main`）、`round22_aggregate.py` |
| `data/result/paper_extra/Round23/` | 第23弾の確認（`check/`）、照合する視界（`views_r20/` 151 MB・`views_dir/` 20 MB）、特徴と本当の読み出し（`ref/`、540 MB）、評価（`eval_r20/` 130 ファイル 161 MB、`eval_dir/` 120 ファイル 30 MB）、集計・表・図（`analysis/`）、試し（`synthetic/`） | 約 920 MB | §47.1〜47.3：`round23_check.py`、`round23_views.py`、`round23_eval.py encode・check・true`、`run_round23_eval.sh <part> <w> 9`（`round23_eval.py main`）、`round23_aggregate.py`、`round23_posthoc.py` |
| `data/result/paper_extra/Round24/` | 第24弾の作り方の確かめ（`test/`）、§1.2 のデータ（`clue/`、位置だけ 3600 エピソード、13 MB）、画像用のエピソード（`check/`）、モデルの確かめ・時間（`model/`）、§2 の評価（`eval/` 12 ファイル）、表・図（`analysis/`） | 約 30 MB | §48：`simulation/collect_data_round24.py`、`round24_model.py`、`round24_clue.py`、`round24_eval.py run・aggregate` |
| `data/data/paper_r25_swap50/` | 第25弾の学習用データ（`data.h5`、train 3000・test 300、101 フレーム、入れ替え 50%・時点 20〜80・`p_rule` 0.5） | 2.0 GB | §49.1：`simulation/collect_data_round24.py --key paper_round25_swap50 --split train --p_rule 0.5 --swap_t rand20-80 --p_swap 0.5 --n_ep 3000 --part <i> --n_parts 10`、test は `--key paper_round25_swap50_test --split test --n_ep 300 --n_parts 2`、`round25_make_data.py` |
| `data/result/paper_extra/Round25/` | 第25弾の試し（組1）：走らせたもの 12 本（`runs/`、重み 5 エポックごと・評価 `eval/`）、確かめ（`check/`）、データの分割（`data_parts/`）、記録（`logs/`）、集計・表・図（`analysis/`） | 571 MB | §49：`round25_train.py check`、`run_round25.sh <cond>`（`round25_train.py stage1・stage2`）、`round25_aggregate.py` |
| `data/result/paper_extra/Round26/` | 第26弾：作りの確かめ（`check/`）、§1 の窓の損失（`signal/`、4 条件 × 6 組）、§2 の走らせたもの 18 本（`runs/s2_*`、足した層の重み・評価 `eval/`）、§3.2 の 2 本（`runs/s32_*`）、§3.1（`data31/`）、記録（`logs/`）、集計・表・図（`analysis/`） | 62 MB | §50：`round26.py check`・`signal --jobs <条件>:<組>`・`train2 --cond --flow --pair`・`data31`・`train32 --ratio`、`round26_aggregate.py sig`・`s2`・`s3` |
| `data/result/paper_extra/Round27/` | 第27弾：作りの確かめ（`check/`）、§1 の窓の損失を領域で分けたもの（`signal/`、4 条件 × 6 組）、§2 の走らせたもの 9 本（`runs/s2_*`、足した層と共有の LSTM の重み・評価 `eval/`）、記録（`logs/`）、集計・表・図（`analysis/`） | 57 MB | §51：`round27.py check`・`sig --jobs <条件>:<組>`・`train --cond --pair`、`round27_aggregate.py sig`・`s2` |
| `data/result/paper_extra/Round28/` | 第28弾：作りの確かめ（`check/`）、§1 の手順1 の 9 本（`stage_a/runs/s2_*_B_s3〜s5`、第26弾と同じ形）、手順2 の 9 本（`runs/mix_*`）、§2 の 6 本（`runs/nomix_p50_s*`）、足した層と共有の LSTM の重み・評価 `eval/`、記録（`logs/`）、集計・表・図（`analysis/`） | 77 MB | §52：`round28.py check a`・`check bc`・`stage_a --cond --pair`・`train --cond --pair --mix 0/1`、`round28_aggregate.py s1`・`s2`・`all` |
| `data/result/paper_extra/Round29/` | 第29弾：§1 の照合の視界と特徴（`views/`・`ref/`）、作りの確かめ（`check/`）、§1 の評価 120（`s1/`）、§2（`s2/`）、§3 18（`s3/`）、§4（`s4/`）、記録（`logs/`）、集計・表・図（`analysis/`） | 115 MB | §53：`round29.py views`・`encode`・`s1 --worker`・`s2`・`s3`・`s4`・`check`、`round29_aggregate.py all` |
| `data/result/paper_extra/Round30/` | 第30弾：作りの確かめ（`check/`）、§1 の網と十分統計量（`s1/`）、新しい復号器（`dec/`）、§2 の 18（`s2/`）、§3（`s3/`）、記録（`logs/`）、集計・表・図（`analysis/`） | 44 MB | §54：`round30.py check`・`s1`・`dec`・`s2 --cond`・`s3 --part`、`round30_aggregate.py` |
| `data/result/paper_extra/Round31/` | 第31弾：作りの確かめ（`check/`）、格子の全配置の特徴・画素の差・範囲ごとの量（`main/grid.npz`）と例の画像（`main/examples.npz`）、§2.4 の読み出し（`read/`）、記録（`logs/`）、集計・表・図（`analysis/`） | 290 MB | §55：`round31.py check`・`main`・`read`、`round31_aggregate.py` |

- 学習に使ったほかのチェックポイント（経験なし・経験50%・経験80%・符号無作為（経験80%）・好み切替）は、§35 より前の作業（v12〜v14、第6弾）のもので、それぞれの作業記録にある。
- 作り直しの手順は、どれも作業記録の該当の節に、コード・鍵・コマンドとして書いてある。
  - 例外：E1 の学習の GPU での再現性は確かめていない。
  - 例外：指示書の原本は git に入っていない。
