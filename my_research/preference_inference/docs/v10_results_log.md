# v10 数値の一次記録

追記専用（行を消さない）。各行に：日付、段、実験ID（config・シード・エポック）、git commit、学習・評価のシード、データセットと split、配置（元／入れ替え／混在）、値（late-5 平均 ± チェックポイント間 sd）、収束判定（`min(B)` 併記）、出力ファイル、図の目視確認の有無。解釈は `v10_experiment_log.md` に書く。

## 1. 環境・資産の確認

| 日付 | 項目 | 値 | commit | 備考 |
|---|---|---|---|---|
| 2026-10-15 | ディスク空き（/home） | 146 GB / 894 GB（84% 使用） | 11477f0 | `df -h` |
| 2026-10-15 | GPU | GPU0 他ユーザー（1.3GB）、GPU1〜7 空き | 11477f0 | `nvidia-smi` |
| 2026-10-15 | 鏡映対称 \|緑(x,y)−赤(x,−y)\| 最大（7×7 格子、元の配置、他者退避） | 0.0 | 11477f0 | 読み取り専用の描画テスト |
| 2026-10-15 | 4隅での色の割合（A-1 を (−9,9)/(−9,−9)/(9,−9)/(9,9) に置く） | 赤 0.0938 / 緑 0.0938 / 青 0.0938 / シアン 0.0938（各地点で最大の色） | 11477f0 | 同上 |

## 2. 関門 6.0（2026-10-15）

実行時の HEAD は `f63b69b`（v10 のコードは未コミットの作業ツリー。次のコミット `6.0 完了` に含む）。シミュレータ `sim_v10/`、GPU1、コンテナ内 xvfb。

| 日付 | 確認 | 値 | 判定 | 出力 | 目視 |
|---|---|---|---|---|---|
| 2026-10-15 | 1. `original` で `r2_a1random_a2rl` の train 先頭 3 ep を同じシード（md5 `r2_a1random_a2rl` → 2205419040）・同じ A-2 actor（`v3_rl_a2_actor.pth`）で再収集 | `self_vision`・`other_vision`・`self_motion`・`self_position`・`other_motion`・`other_position` の6配列すべて `array_equal=True`、max\|diff\|=0 | **合格** | `data/result/v10_swap/gate60/bitcheck_original.h5`（git 管理外） | — |
| 2026-10-15 | 2. 21×21 格子（A-2 は (50,50)）で \|赤(swap)−緑(orig)\|、\|緑(swap)−赤(orig)\|、\|青(swap)−青(orig)\|、\|シアン(swap)−シアン(orig)\| の最大 | 0.0 / 0.0 / 0.0 / 0.0（画素数で 0） | **合格** | `gate60/gate60_layout_check.json`、`gate60_fractions.npz` | — |
| 2026-10-15 | 配置を `swap_rg` にして `original` に戻したときの視界（441 点） | 全点で bit-identical | 合格（追加の確認） | 同上 | — |
| 2026-10-15 | 3. 報酬マップ（4色の割合・A-1 の報酬、両配置） | A-1 の報酬の範囲は両配置とも [−0.164, +0.164]。入れ替えで赤・緑の割合と報酬が上下に反転、青・シアンは不変 | 図を作成 | `gate60/gate60_fraction_reward_maps.png` | **目視確認済み** |
| 2026-10-15 | 4. swap の (x,y) の視界と original の (x,−y) の視界（§4.2 の確認） | 441 点中、一致する点 **0**。異なる画素数 最小 18・平均 359.4。赤・緑の割合の差は最大 0.0029（3 画素、格子の端の点での描画の非対称） | **一致しない（確認どおり）** | `gate60/gate60_view_comparison.png`、`gate60_view_examples.npz` | **目視確認済み** |

### 2.1 P2：`test_v10.py` の出力先の確認（2026-10-15、HEAD `f63b69b`）

- `analyze/v10/test_v10.py --exp_config r3_a_direct_1000pretrain_400 --seed 0 --test_epoch 400 --test_data_name r2_a1random_a2rl_viz20 --test_modes eval --eval_seed 0 --cudnn_deterministic --save_targets state self_vision other_vision self_position other_position`、GPU1、`--out_dir data/result/v10_swap/p2_check/run_{a,b}` で2回。
- 実行前後で `data/result`（`v10_swap` を除く）・`data/model`・`data/data` の全ファイル・ディレクトリのパス・サイズ・更新時刻（7738 行）を比較 → **差分なし**（保護対象の中に新しいファイル・ディレクトリは作られていない）。
- 2 回の `saved.h5` の全 13 データセットが **bit-identical**。`self_vision/{input,truth,prediction}`、`other_vision/{input,truth}`、`state/{self,other}/{hidden,cell}`、位置が保存されることを確認。
- 確認後、`p2_check/` の `saved.h5`（2 本、計 243MB）は削除した（このファイルから作る図・JSON は無い）。

## 3. 段階A

### 3.1 準備の確認（2026-10-15、HEAD `ad029a0` + 未コミットの v10 スクリプト）

| 項目 | 値 | 出力 |
|---|---|---|
| 対のシードの確認（swap、`r2_a1random_a2rl` のキー、3 ep） | `self_position`・`self_motion` bit-identical、A-2 の t=0 位置一致、A-2 の位置の差は各 ep 最大 1.01 / 1.01 / 0.96 | `stageA/pairseed_check_swap.h5`（管理外） |
| `r2_a1random_a2rl_viz20` = `r2_a1random_a2rl` train[:20] | 全6配列一致 | — |
| grid 収集スクリプトの確認（original、外側の 3 位置 × 441） | root `grid` の round(v·255) と 4 視界すべて一致、位置一致 | （確認後に削除） |
| `test_v10.py` で R3-A_400 s0 ep400、A0、eval（train split） | h¹→自己 0.908456（v4 の `baseline_v4/..._s0_ep400_..._r2.txt` 0.9085 と一致）。self_vision 損失 9.5066。所要 4分27秒（eval+test） | `stageA/eval/r3_a_direct_1000pretrain_400/s0/ep400/A0/` |
| Fig.5 `fig5_v4_exp1l1000` ep100、元の grid、margin 1 | self_self 0.04572 / other_other 0.05806 / self_other 0.13571 / other_self 0.14915、マージン 0.0911（v4 の 0.0457 / 0.0581 / 0.1357 / 0.1491 を再現） | `stageA/fig5/fig5_v4_exp1l1000_ep100_A0.json` |

### 3.2 critic・actor の最良方向マップ（2026-10-15、HEAD `ad029a0`、`analyze/v10/direction_maps_v10.py`）

21×21 格子、地図を描くエージェントを格子上で動かし、他方を (0,0) に固定。角度差は基準位置 (±9,±9) から 2 以内を除いた 419 点の平均（括弧内は中央値）。「色」＝その配置での好きな色の方向、「場所」＝もう一方の配置で好きな色があった場所の方向。

| エージェント | 地図 | 配置 | 色への角度差 | 場所への角度差 | 読み | 方向の円周分散 |
|---|---|---|---|---|---|---|
| A-1（赤） | critic 最良方向 K=64 | original | 27.1°（24.1） | 111.3°（105.4） | — | 0.081 |
| A-1 | critic K=64 | **swap_rg** | **110.9°**（103.6） | **26.5°**（22.6） | **場所** | 0.083 |
| A-1 | critic K=8 | swap_rg | 110.6° | 26.6° | 場所 | 0.092 |
| A-1 | actor tanh(mean) | swap_rg | 105.8° | 25.2° | 場所 | 0.080 |
| A-2（緑） | critic K=64 | original | 32.7°（29.7） | 99.7°（95.7） | — | 0.296 |
| A-2 | critic K=64 | **swap_rg** | **104.9°**（98.4） | **31.6°**（30.2） | **場所** | 0.225 |
| A-2 | critic K=8 | swap_rg | 107.1° | 31.6° | 場所 | 0.179 |
| A-2 | actor tanh(mean) | swap_rg | 108.3° | 25.3° | 場所 | 0.102 |

- 参考（基準を立方体の中心 (±13,±13) にした場合、441 点）：A-1 critic swap 113.8° vs 25.5°、A-2 critic swap 113.4° vs 32.7°。読みは同じ。
- 入れ替えによる同じ格子点での最良方向の変化の平均：A-1 critic 1.5°、A-1 actor 3.7°、A-2 critic 8.1°、A-2 actor 1.5°。
- Q(s, a_probe)（8 方向 × 441 点）の両配置間の相関：A-1 0.996（平均の差 0.065、Q の sd 0.959）、A-2 0.905（平均の差 0.074、sd 0.321）。Q 値マップ（8 方向の最大）の最大位置：A-1 は両配置とも (−8,10)、A-2 は両配置とも (−6,−9)。
- 図：`data/result/v10_swap/stageA/direction_maps/direction_maps_A1.png`、`direction_maps_A2.png`（**目視確認済み**：両配置で Q 値マップと矢印がほぼ同じ。A-1 は入れ替え後も左上（今は緑）の価値が高く、北〜北西を向く。A-2 は入れ替え後も左下（今は赤）を向く）。数値 `direction_maps.json`、配列 `direction_maps.npz`。

### 3.3 A-2 の actor の終端位置（2026-10-15、`analyze/v10/a2_terminal_v10.py`）

A-2 = `v3_rl_a2_actor.pth`（元の配置で緑好きに学習）。`r2_a1random_a2rl`（original）と `v10_swap_r2`（swap、同じエピソードのシード）。終端 = t=100 の `other_position`。

| split | 配置 | 終端→左上 (−9,9) 平均（中央値） | 終端→左下 (−9,−9) 平均（中央値） | 左上の方が近い割合 | 終端が左上から5以内 | 終端が左下から5以内 | 全フレームで左上5以内 / 左下5以内 |
|---|---|---|---|---|---|---|---|
| train 3000 | original（緑=左下） | 19.14（19.13） | 2.50（2.45） | 0.000 | 0.000 | 1.000 | 0.002 / 0.905 |
| train 3000 | **swap_rg（緑=左上）** | 19.09（19.08） | **2.07（1.99）** | **0.000** | **0.000** | **1.000** | 0.002 / 0.905 |
| test 300 | original | 19.14（19.13） | 2.47（2.44） | 0.000 | 0.000 | 1.000 | 0.002 / 0.903 |
| test 300 | swap_rg | 19.09（19.08） | 2.03（1.97） | 0.000 | 0.000 | 1.000 | 0.002 / 0.903 |

- 対にしたエピソード（初期位置は全エピソードで一致）：終端位置の差 平均 0.48（train）/ 0.50（test）、全フレームの位置の差の平均 0.42 / 0.42。
- 図：`data/result/v10_swap/stageA/a2_terminal/a2_terminal.png`（**目視確認済み**：両配置とも約 20 ステップで左下の壁際 (x≈−7, y=−10) に着き、距離の推移はほぼ重なる。入れ替え後は左下が「赤」になっているが、A-2 はそこへ行く）。数値 `a2_terminal.json`。

### 3.4 予測画像（§7.1、2026-10-15、`analyze/v10/pred_images_v10.py`）

viz20（同じエピソード番号が両配置で対）、エピソード 0・1・2、t = 1, 5, 10, 20, 50。値は viz20 全 20 エピソード・全時刻の平均絶対誤差（[0,1] 画素値）。

| モデル | Fig.5 Decoder | 配置 | 自己の予測 \|pred−truth\| | 他者（Fig.5 の経路）\|rec−A-2 の視界\| |
|---|---|---|---|---|
| R3-A_400 s0 ep400 | fig5_v4_exp1l1000 ep100 | original | 0.0455 | 0.1150 |
| R3-A_400 s0 ep400 | 同 | swap_rg | 0.0737 | 0.1879 |
| R3-A_400 s1 ep400 | 同 | original | 0.0461 | 0.1150 |
| R3-A_400 s1 ep400 | 同 | swap_rg | 0.0747 | 0.1879 |
| v5_base_l1 s0 ep200 | fig5_v5_base ep100 | original | 0.1067 | 0.2055 |
| v5_base_l1 s0 ep200 | 同 | swap_rg | 0.1373 | 0.1942 |

- 図：`data/result/v10_swap/stageA/pred_images/<model>/pred_images_{original,swap_rg}_ep{0,1,2}.png`、配列 `pred_images_arrays.npz`、数値 `pred_images_summary.json`。**R3-A_400 s0 の ep0 を目視確認済み**：入れ替え後も、自己の予測は赤を右側（元の赤の列）、緑を左側に描く（入力・正解は逆）。Fig.5 の経路の出力も同じく元の配色で描く。
- Autoencoder の自己再構成（Fig.5 の self 経路、現在のフレームをそのまま入力）の赤・緑の画素の左右の割合（viz20、全フレーム）：

| Decoder | 配置 | 入力 赤 左/右 | 再構成 赤 左/右 | 入力 緑 左/右 | 再構成 緑 左/右 |
|---|---|---|---|---|---|
| fig5_v4_exp1l1000 | original | 0.000 / 0.099 | 0.000 / 0.068 | 0.057 / 0.000 | 0.062 / 0.000 |
| fig5_v4_exp1l1000 | **swap_rg** | **0.055 / 0.000** | **0.000 / 0.062** | **0.000 / 0.099** | **0.061 / 0.000** |
| fig5_v5_base | original | 0.000 / 0.099 | 0.000 / 0.038 | 0.057 / 0.000 | 0.042 / 0.000 |
| fig5_v5_base | swap_rg | 0.055 / 0.000 | 0.000 / 0.033 | 0.000 / 0.099 | 0.044 / 0.000 |

### 3.5 Encoder と probe-Q は配置を区別するか（**探索的。事前登録なし**、2026-10-15、`analyze/v10/diag_encoder_colour.py`）

viz20 の対のフレーム（各配置 2020 フレーム）。「対の距離比」＝ 対のフレーム（同じ位置、赤緑だけ違う）の特徴の距離 ÷ 同じ配置の別フレームどうしの距離。「配置の読み取り」＝ロジスティック回帰（標準化、C=1.0、エピソード単位 5 分割、シード 0）の正答率（chance 0.5）。

| モデル | 特徴 | 次元 | 対の距離比 | 配置の読み取り |
|---|---|---|---|---|
| R3-A_400 s0 ep400 | sv_enc（自己 Encoder、元論文のもの・凍結） | 64 | 0.121 | 0.879 |
| R3-A_400 s0 ep400 | ov_enc（他者 Encoder） | 64 | 1.104 | 0.983 |
| R3-A_400 s0 ep400 | probe-Q（A-1 critic、8 方向） | 8 | 0.102 | **0.524** |
| v5_base_l1 s0 ep200 | sv_enc（価値入力で再学習） | 64 | 0.569 | 0.899 |
| v5_base_l1 s0 ep200 | ov_enc | 64 | 0.692 | 0.924 |
| （共通） | 画素 | 3072 | 0.627 | 1.000 |

出力：`data/result/v10_swap/stageA/diag/encoder_colour_*.json`。

### 3.6 4軸 R² と視覚予測誤差（late-5、2026-10-15、`analyze/v10/run_stageA_eval.sh` → `stageA_regression.py` → `aggregate_stageA.py`）

評価：`test_v10.py`、マスク on（`p_mask_vision` 0.99）、`--eval_seed 0`、cudnn deterministic。train split = `eval` グループ（3000 ep、in-sample の Ridge α=1）、test split = 300 ep。A0 = `r2_a1random_a2rl`、A1 = `v10_swap_r2`（A-1 の道筋は A0 と同一、確認済み）。値は late-5 平均 ± チェックポイント間 sd。HEAD `ad029a0`（スクリプトは次のコミットに含む）。

**R3-A_400**（seed0・seed1、ep360〜400。v4 の収束判定：両シード収束）

| 指標 | A0 | A1 (i) 再 fit | A1 (ii) A0 の回帰器 | held-out（A0 の回帰器）A0 test / A1 test |
|---|---|---|---|---|
| h¹→自己 | 0.9065（s0 0.9060±0.0017 / s1 0.9069±0.0034） | 0.8938（0.8933±0.0012 / 0.8943±0.0027） | 0.8896（0.8890 / 0.8902） | 0.8916 / 0.8813 |
| h¹→他者 | 0.5777（0.5820 / 0.5735） | 0.5821 | 0.5551 | 0.5583 / 0.5374 |
| h²→自己 | 0.0879 | 0.0896 | 0.0478 | 0.0911 / 0.0542 |
| h²→他者 | 0.9588（0.9588±0.0018 / 0.9589±0.0008） | 0.8651（0.8651±0.0015 / 0.8651±0.0022） | 0.7670（0.7665 / 0.7675） | 0.9533 / 0.7518 |
| 自己の視覚予測誤差（eval ループの L1、時刻の和）train / test | 9.616 / 9.999 | 14.365 / 14.478 | — | — |

- A0 は v4 の値を再現（h¹→自己 0.9065 = `thesis_skeleton` の 0.9065、h²→他者 0.959 ≈ `results_log` の 0.957）。
- A1/A0 の比（(i)）：h¹→自己 0.986、h²→他者 0.902。視覚予測誤差 +49%（train）。

**v5_base_l1**（seed0、ep160〜200。Decoder のみ学習の段なので、共有モジュール・Encoder は 5 チェックポイントで同一 → R² の sd は 0）

| 指標 | A0 | A1 (i) | A1 (ii) | held-out A0 test / A1 test |
|---|---|---|---|---|
| h¹→自己 | 0.7175 | 0.7216 | 0.7153 | 0.7121 / 0.7116 |
| h¹→他者 | 0.4960 | 0.5076 | 0.4718 | 0.4829 / 0.4587 |
| h²→自己 | 0.0810 | 0.0767 | 0.0632 | 0.0804 / 0.0612 |
| h²→他者 | 0.6455 | 0.5910 | 0.5142 | 0.6256 / 0.5087 |
| 自己の視覚予測誤差 train / test | 21.680±0.194 / 21.932 | 26.191±0.264 / 26.334 | — | — |

- A0 の h²→他者 0.6455 は `thesis_skeleton` の v5 共通評価 0.645 と一致。
- 図：`data/result/v10_swap/stageA/eval/{r3_a_direct_1000pretrain_400,v5_base_l1}/stageA_r2_bars.png`、`stageA_vision_error_bars.png`（**目視確認済み**）。数値 `stageA_late5_summary.json`、チェックポイントごと `s<seed>/ep<E>/stageA_regression.json`。中間 `saved.h5` は各 JSON の作成後に削除。

### 3.7 Fig.5（2026-10-15、grid、margin 1、Decoder ep100、`analyze/v10/vpt_metrics_v10.py`）

| Encoder / Decoder | 条件 | self_self | other_other | self_other | other_self | マージン（other_self − other_other） |
|---|---|---|---|---|---|---|
| 元論文（R3-A_400 が使う）/ fig5_v4_exp1l1000 | A0 元の grid | 0.0457 | 0.0581 | 0.1357 | 0.1491 | 0.0911 |
| 同 | **A1 入れ替えの grid** | 0.0771 | **0.1075** | 0.1366 | 0.1594 | **0.0519** |
| v5 / fig5_v5_base | A0 | 0.0722 | 0.1335 | 0.1298 | 0.1584 | 0.0249 |
| 同 | A1 | 0.1061 | 0.1416 | 0.1423 | 0.1730 | 0.0314 |

- 両方の Encoder で、入れ替え後も other_other < other_self（自他の向きは保たれる）。元論文の Encoder では other_other がほぼ倍になり、マージンが 0.091 → 0.052。
- 図：`data/result/v10_swap/stageA/fig5/fig5_bars.png`（**目視確認済み**）。grid の `saved.h5`（各 6.7GB）は JSON 作成後に削除。
