# 方法の節（`paper/notes/methods.tex`）の事実確認（指示書 §2 の項目 1・2・4〜14）

- 2026-10-26。コード・設定・ログ・保存済みの結果を読むだけで確かめた（学習・GPU なし。チェックポイントの比較はコンテナの CPU）。
- 項目 3（方角だけの基準の BA）は別に計算する（`paper_round1_log.md` §1、`direction_baseline.json`）。
- パスは `my_research/preference_inference/` から。このファイルは内部の確認用なので版の番号も使う。

---

## A. 本文と違う点・直した方がよい点（先に直すもの）

| # | 本文の箇所 | 本文の記述 | 正しいこと | 根拠 |
|---|---|---|---|---|
| A1 | 3.3 段階I（項目 2） | 「後半の 200 エポック（L1 期間）は絶対誤差で視覚予測器だけを学習した」。式では $\hat v_{t+1}=\phi_{\mathrm{pred}}(h^1_t,h^2_t)$ を視覚予測器と呼んでいる | **L1 期間に変わるのは復号部 `vision_decoder_module`（全結合 1 層＋転置畳み込み 3 層）だけ**。h¹・h² をまとめる層は別のモジュール `integration_module`（`cat[h¹,h²]`（各 128、学習時は dropout 0.5）→ 全結合 256→64 → ReLU）で、**MSE 期間だけ学習し、L1 期間は凍結**。したがって $\phi_{\mathrm{pred}}$ を「視覚予測器」と呼ぶなら、L1 期間に学習するのは「視覚予測器のうち復号部（統合層の後ろ）」と書く必要がある。文献の元のコードの exp1_l1 も同じ凍結（`integration_module` を凍結） | 設定 `config/exp/v12_s1_mixed_l1_s0.yml`（freeze：superposition_module・vision_encoder_module・integration_module・share_lns・feature_prediction_module。v12/v13/NS の l1 の設定は全組でデータ名・シードだけが違う）。学習ログ `data/result/v12_s1_mixed_l1_s0/train_seed0.log` の「Train params」は `vision_decoder_module.*` の 8 個だけ。チェックポイントの比較（CPU）：`v12_s1_mixed_l1_s0` の ep160 と ep200、mse ep200 と l1 ep160 で値が変わるキーは `vision_decoder_module.{fcs.0, conv.convs.0〜2}.{weight,bias}` の 8 個だけ（44 キー中）。モデル `model/model_v12.py` の forward（`integration_module(dropout(ss), dropout(os))` → `vision_decoder_module`）、`model/modules.py:109-121`、`config/model/SuperpositionNetworkMotionPleasureHold/default.yml`（integration_module：each_input 128、output 64）。元のコード `/home/kusano/superposition/config/exp/exp1_l1_1000.yml` |
| A2 | 3.2 基準エンコーダ（項目 10） | 「基準エンコーダを用いた再現モデルでは，h¹ から A-1 の位置が R²=0.977，h² から A-2 の位置が 0.973」 | 0.977・0.973 は **`exp1_l1`（100 エピソードの元論文の設定）** の値（h¹→self 0.9772、h²→other 0.9725）。**基準エンコーダの元のモデル `exp1_l1_1000`（1000 エピソード、本研究で凍結に使った重み）では h¹→self 0.9667、h²→other 0.9932**。なお 0.9725 を 3 桁にすると 0.972 か 0.973（丸め方による）。どちらも「旧」の評価（P0-0 前、非決定的な評価・シード 0 のみ、Ridge）。図5に当たる取り出しのマージン +0.091 と自他一致度 0.80（0.799）は `exp1_l1_1000` の値で合っている | `docs/v4_experiment_log.md` §5.1（exp1_l1 の行）・§8.4（exp1_l1_1000 の行、839・875 行）、`docs/results_log.md` 67・68 行。凍結に使った重みは `pretrain_encoder_from: /work/data/result/exp1_l1_1000/0/model/00200.pth`（`config/exp/v12_s1_*_mse.yml`・`v13_s1_mixed_mse.yml`・`v13ns_s1_*_mse.yml`）。v12 本番の mse ep200 の Encoder・share_lns の 22 キーが `exp1_l1_1000/0` ep200 と完全一致することを CPU で確かめた |
| A3 | 3.4 判定の共通の規則（項目 13・14 に関わる） | 「経験ありの値が仮説の向きで，かつ組の差の信頼区間が0を含まない点が5点中3点以上」 | **G3・G5a・G5b・色の基準では、経験ありの値も CI が 0 を含まないこと**（仮説の側）を求めている。実験4（BA）は「経験ありの BA の CI の下限 > 0.5 かつ組の差の CI の下限 > 0」。本文の書き方（経験ありは向きだけ）に当たるのは G2 の M1(c)・M1(d) だけ（経験ありは平均の符号、組の差は CI） | `analyze/v12/aggregate_s1_v12.py:59-61`（M1d：平均 > 0 かつ組の差の CI、colour：CI、G3：経験ありの CI と組の差の CI）、`analyze/v12/aggregate_g5_v12.py:36,44`（G5a・G5b：両方の CI）、`docs/v12_experiment_log.md` §17・§13、`docs/v13_experiment_log.md` §31 の 3、`docs/v14_experiment_log.md` §3 D |
| A4 | 3.1 データ／3.4 実験2・3（緑好みデータ） | 「A-2 が常に緑を好むデータ（緑好みデータ）…を，それぞれ学習用 3000・評価用 300 エピソード作った」 | 学習は `v11_s2_green`（train 3000、シードの鍵 `v11_s2`）だが、**評価（T2、T3、G5a・符号に従う分の ①）に使った緑好みの test は別に集めた `v11_s2t_green`（test 300、鍵 `v11_s2t`）**。`v11_s2_green` にも test 300 があるが評価には使っていない。T2 は `v11_s2t_green` と `v11_s2t_green_swap`、T5 は `v11_s2t_green` と `v11_s2t_red`（赤好き）の test を組にして作る（好み混合データではない）。本文の「評価用データ」が `v11_s2t_*` であることを一文足すとよい（対応表 §7 には書いてある） | `analyze/v11/eval_s2_v11.py:39-40`（SETS：green = `v11_s2t_green` test、swap、red、pref = `v11_s2_pref` test）、`analyze/v11/t1t3_s2_v11.py:31`（T3 は green・pref）、`simulation/collect_data_v11_s2t.py:1-55` |
| A5 | 3.4 実験1〜4（手順の違い） | G5b は「t=0〜19 は通常どおり流し」、符号に従う分・T3・実験4 は流し方を書いていない | 流し方が二通りある。**G3・G5b**：段階Iと同じ（視覚は確率 0.99 で隠す、運動は自分の隠す区間のとおり、`util.seed_all(0)`）。**G5a／符号に従う分、T3（真値条件）、実験4、段階IIの評価**：段階IIと同じ（**視覚は t=0 だけ見せ t>0 は process-1・2 とも隠す、process-1 の運動は隠さない（印 0、今の快・不快）**）。本文に一文足すとよい | `analyze/v12/eval_g5_v12.py:202,222`（G5a は `motion_masking=False`、G5b は `True`）・`:87`（G5a は t>0 で視覚 p=1）・`:130-150`（G5b は `p_mask`）、`analyze/v12/eval_trial_v12.py:137-156`（G3）、`docs/v14_experiment_log.md` §2.1（実験4） |
| A6 | 3.3 段階II | 「損失は A-1 の次の視界の絶対誤差と特徴予測の誤差で，後者の勾配は h² を通って G に届く」 | 誤りではないが不足：**絶対誤差（視界）の勾配も h² → 統合層 → 復号部を通って G に届く**（段階IIの runner は特徴予測の入力を止めず、統合層は h² を使う）。process-1 側の特徴予測（h¹）は G によらない | `exp/runner.py:181-197`（`MotionGenerationPredictionFeaturePredictionRunner`：`predict_feature(os)` を detach しない、損失 = l1(self_vision) + FP_self + FP_other）、`model/model_v12.py:154-167` |
| A7 | 3.2 重ね合わせ機構（小さな注記） | 「同じ作りで重みの異なる2つの視覚エンコーダ（…層正規化）」 | 畳み込み・全結合の重みは2つで異なる（確認済み）が、**層正規化（`share_lns`）は2つのエンコーダで共有**（文献の元のコードと同じ）。凍結なので結果には関わらないが、「層正規化は2つで共有」と書くと正確 | `model/model.py:14-22`（`add_vision_encoder_module`：`share_lns` を両方に渡す）。v12 mse ep200 で `self_` と `other_vision_encoder_module` の conv・fc の重みが異なることを CPU で確認 |
| A8 | 3.3 段階II の 1ビットモジュール | 「$c^{+}$ は…段階Iの学習データの $|P|$ の範囲ごとの平均から作った」 | 値は合っている。作り方の正確な言い方：**経験50%条件（`v11_s1_mixed`）と経験なし条件（`v11_s1_random`）の学習データ（train 3000 × t=0〜99）の |P| の範囲ごとの平均を二つの条件で平均し、範囲 k と 7−k を平均して左右対称にした**（経験80%条件のデータは使っていない） | `docs/v14_experiment_log.md` §2.2、`config/model/SuperpositionNetworkMotionPleasureHoldOtherOneBit/onebit.yml`（shape_vector [−0.1041654, −0.12710697, 0, 0, 0, 0, 0.12710697, 0.1041654]） |
| A9 | 3.4 実験2 | 「予測画像からの A-2 の位置は，背景より暗い画素の列ごとの量から方角と距離を求めて読み取った」 | 主の読み取り（G5a・G5b の判定に使ったもの）は「**黒に近い画素**（各画素の RGB の最大が 0.2 未満の分を重みにする）の列ごとの量」から方角（列の角度の円周上の重心）と距離（本当の視界の黒い画素の数と距離の対数の直線で合わせる）を求める。「背景より暗い画素」（行の中央値より暗い分）は副の読み取り（B1）の定義 | `analyze/v11/p2test_v11.py:35-55`（`calibrate`・`read_a2`、MIN_MASS 2.0）、`analyze/v12/eval_g5_v12.py:38-45`（`bg_weights`、B1）、`docs/v12_experiment_log.md` §22 |
| A10 | 3.1 経験の量（言い方） | 「経験80%条件の A-1 は中央値14ステップで赤の角に着き」 | 数え方は「**赤から距離 5 以内に最初に入る時刻**の中央値 14」（50% 35、経験なし 101＝入らない）。「角に着く」は v13 まとめの言い方で、数えた量は「距離 5 以内に入る」 | `docs/v13_results_log.md` §1（最初に 5 以内に入る時刻の中央値 101 / 35 / 14）、`docs/v13_summary.md` §2 |

- 補足（本文の主張には関わらない）：s4・s5 にも「真値条件に近い量」がある。`data/result/v13/main/d6/s{0..5}/d6_{random,mixed}_ep200.json` と `v13/v12ref/d6/s{4,5}/d6_mixed_ep200.json`（段階Iの D6 に当たる副、`analyze/v12/eval_d6_v12.py`、process-2 = [0, P_A2, 1] の他者側の特徴予測の誤差 `B.fp`、ep200 だけ）。ただし T3 の D6（`stage2/t1t3`）とは少し違う値になる（例 s0 の 80%・緑好み：T3 0.4017、こちら 0.4051。手順の細部が違うと考えられるが、原因は確かめていない）。本文の「実験2（真値条件の評価を除く）」は T3 の D6 についてなら合っている。

---

## B. 項目ごとの確認

### 項目 1：段階I本番の最適化の設定

| 本文の記述 | 判定 | 正しい値・補足 | 根拠 |
|---|---|---|---|
| AdamW | 合っている | | `config/exp/v12_s1_{random,mixed}_mse.yml`・`v13_s1_mixed_mse.yml`・`v13ns_s1_{mixed,random}_mse.yml`、l1 段の `*_l1_s{0..5}.yml` |
| 学習率 $10^{-3}$ | 合っている | 0.001（MSE 期間・L1 期間とも） | 同上 |
| 重み減衰 0.01 | 合っている（補足あり） | 既定 0.01。**名前に `bias` を含むパラメータ（LSTM の bias_ih・bias_hh、全結合・畳み込みの bias、層正規化の bias）は 0.0**。設定には `vision_encoder`・`share_lns` に 0.001 の指定もあるが、どちらも凍結なので使われない。特徴予測器の層正規化の weight は 0.01 | 同上の `per_params`、`util.py:180-229`（名前の部分一致で振り分け） |
| バッチ 10 | 合っている | | 同上 |
| 200 エポック＋200 エポック | 合っている | mse 段 max_epochs 200（毎エポック保存）、l1 段 200（10 エポックごと保存） | 同上 |
| 条件の間の違い | — | 経験なし・50%・80%・非共有の mse の設定は、データ名（`v11_s1_random`／`v11_s1_mixed`／`v13_s1_mixed80`）とモデルのクラス（非共有は `...HoldNS`）だけが違う。l1 の設定はデータ名・pretrain 元・シードだけが違う（全 6 組で確認） | `diff` で確認 |
| 視覚のマスク 0.99 | 合っている | `p_mask_vision: 0.99`、t=0 は見せる（`PredictionRunner.gen_mask_prob`） | 設定、`exp/runner.py:146-148` |

### 項目 2：L1 期間に学習する範囲 → **A1**

- MSE 期間に学習するもの：`superposition_module.lstm`（非共有は `lstm_other` も）、`integration_module.fc`、`vision_decoder_module`、`feature_prediction_module`（学習ログ `data/result/v12_s1_mixed_mse/train_seed0.log`・`v13ns_s1_mixed_mse/train_seed0.log` の Train params）。凍結：2 つの視覚エンコーダと `share_lns`。
- L1 期間に学習するもの：`vision_decoder_module` だけ（ログ・チェックポイントの比較で確認）。
- 特徴予測器：入力（h¹・h²）と目標（次の視界のエンコーダ出力）を detach しており、段階Iの特徴予測の損失は特徴予測器だけを学習する（`exp/runner.py:154-168`）→ 本文の「特徴予測器への入力と目標は勾配を止めてあり」は**合っている**。

### 項目 4：組5・6（s4・s5）のモデル名とチェックポイント

| 条件 | MSE 期間 | L1 期間 | 判定 |
|---|---|---|---|
| 経験なし | `data/result/v12_s1_random_mse/{4,5}/model/00000〜00200.pth`（201 個） | `data/result/v12_s1_random_l1_s{4,5}/{4,5}/model/00000〜00200.pth`（21 個） | 対応表の規則どおり |
| 経験50% | `v12_s1_mixed_mse/{4,5}/` | `v12_s1_mixed_l1_s{4,5}/{4,5}/` | 同上 |
| 経験80% | `v13_s1_mixed_mse/{4,5}/` | `v13_s1_mixed_l1_s{4,5}/{4,5}/` | 同上 |

- 設定：`config/exp/v12_s1_{random,mixed}_l1_s{4,5}.yml`、`v13_s1_mixed_l1_s{4,5}.yml`（mse は既存の `*_mse.yml` を `--seed 4/5` で）。`docs/v13_experiment_log.md` §31。

### 項目 5：段階IIのモデル名（`data/result/` 直下、各 `<名前>/<シード>/model/` に ep0〜200 の 21 個）

| 論文の名前 | モデル名 | 組 | 設定のクラス |
|---|---|---|---|
| 自由出力モジュール（経験50%・経験なし） | `v12_s2_{green,pref}_{mixed,random}_s{0..3}` | s0〜s3 | `SuperpositionNetworkMotionPleasureHoldOther` / default |
| 自由出力モジュール（経験80%） | `v13_s2_{green,pref}_mixed_s{0..3}`（経験なしは `v12_s2_*_random_s*` を再利用） | s0〜s3 | 同上（pretrain = `v13_s1_mixed_l1_s*`） |
| 形制限モジュール | `v12_s2b_{green,pref}_{mixed,random}_s{0..3}` | s0〜s3 | `...HoldOtherShaped` / signed_pairs |
| 他者遮蔽学習 | `v12_s2c_{green,pref}_{mixed,random}_s{0..2}`（データ `v12_hide_v11_s2_{green,pref}`） | s0〜s2 | `...HoldOtherShaped` / signed_pairs |
| 1ビットモジュール | `v14_s2u_pref_{random,mixed,mixed80,mixedpref}_s{0..2}`（好み混合データだけ） | s0〜s2 | `...HoldOtherOneBit` / onebit |
| 部分制限モジュール（付録） | `v12_s2a_{green,pref}_{mixed,random}_s0` | s0 | `...HoldOtherShaped` / mask |

- `config/exp/v14_s2u_pref_{mixed,random}_s3.yml` は設定だけあり、結果のディレクトリはない（回していない）。

### 項目 6：実験2の真値条件（T3 の D6）に組4（s3）の値があるか → **ある**

他者側の特徴予測の誤差（`data/result/v12/stage2/t1t3/` と `v13/stage2/t1t3/` の `<set>_<cond>_s<s>_ep<e>.json` の `<set>` → `zero`・`d6`・`gen` の `other_fp_mse`。緑好みの値は緑好みデータで学習した G、好み混合の値は好み混合で学習した G のファイル。gen は late-5 の範囲）

| 組 | データ | 経験なし 零 / D6 / 生成 | 経験50% 零 / D6 / 生成 | 経験80% 零 / D6 / 生成 |
|---|---|---|---|---|
| s0 | 緑好み | 0.3147 / 0.2546 / 0.1308〜0.1336 | 0.3025 / 0.2764 / 0.1319〜0.1354 | 0.5451 / 0.4017 / 0.1286〜0.1330 |
| s0 | 好み混合 | 0.3126 / 0.2553 / 0.1464〜0.1510 | 0.3039 / 0.2651 / 0.1328〜0.1376 | 0.5186 / 0.3907 / 0.1119〜0.1300 |
| s1 | 緑好み | 0.3257 / 0.2856 / 0.1587〜0.1765 | 0.3053 / 0.2517 / 0.1165〜0.1218 | 0.3048 / 0.4062 / 0.1214〜0.1263 |
| s1 | 好み混合 | 0.3237 / 0.2850 / 0.1665〜0.1769 | 0.3042 / 0.2484 / 0.1209〜0.1274 | 0.2980 / 0.3960 / 0.1233〜0.1272 |
| s2 | 緑好み | 0.3025 / 0.2620 / 0.1249〜0.1320 | 0.2974 / 0.2545 / 0.1379〜0.1416 | 0.4395 / 0.4179 / 0.1419〜0.1567 |
| s2 | 好み混合 | 0.3032 / 0.2640 / 0.1284〜0.1308 | 0.2931 / 0.2491 / 0.1376〜0.1394 | 0.4127 / 0.4141 / 0.1343〜0.1446 |
| **s3** | 緑好み | 0.3105 / 0.2715 / 0.1413〜0.1447 | 0.3436 / 0.2515 / 0.1400〜0.1460 | 0.3634 / 0.3765 / 0.1260〜0.1341 |
| **s3** | 好み混合 | 0.3085 / 0.2704 / 0.1415〜0.1489 | 0.3487 / 0.2408 / 0.1431〜0.1588 | 0.3539 / 0.3767 / 0.1413〜0.1880 |

- 零と D6 は G によらない（段階Iだけで決まる）ことを確かめた：(1) 同じ G の ep160〜200 の 5 点で値が一つ（上の表、全 4 組・3 条件・2 データ）、(2) 緑好みで学習した G の ep200 と好み混合で学習した G の ep160 で、両方のデータの零・D6 が完全に一致（全組・全条件）、(3) 経験なしの値は v12 と v13 のファイルで完全に一致（同じモデルの再利用）。`docs/v13_experiment_log.md` §24 の確認とも合う。
- s3 の 80% の緑好みの値（生成 0.129 / 零 0.363 / D6 0.376）は `docs/v13_experiment_log.md` §26 と一致（生成は ep200 の値）。
- 一歩の種類ごとの集計（`v13/post/d6_split/`）は s0〜s2 だけ（s3 なし）。
- s4・s5 は段階IIを回していないので T3 はない（上の A の補足の「D6 に当たる副」はある）。

### 項目 7：環境とデータ

| 本文の記述 | 判定 | 根拠 |
|---|---|---|
| 20×20 のアリーナ | 合っている | `sim_v10/config/world/default.yml`（xmin −10、xmax 10、ymin −10、ymax 10） |
| 四隅の外に4色の立方体 | 合っている | プロジェクトのメモ（立方体の中心 (±13, ±13)、v10 の記録）、`v11core/pleasure.py` の docstring（各ランドマークがちょうど 16 列） |
| 101 フレーム | 合っている | `simulation/collect_data_v11.py:185`（seq_length 101）、`collect_data_v11_s2t.py` |
| 休止：確率 0.005、10〜30 ステップ | 合っている | `sim_v10/config/agent/random.yml`（sleep_p 0.005）、`agent/base.yml`（sleep_base 20、sleep_margin 10 → 20 + U{−10..10}）、`v11core/rule_agent.py:96-99`（起きている一歩のあとに毎回判定）、`sim_v10/simulation/agent/agent_base.py` |
| 1ステップに1の距離 | 合っている | `agent/base.yml`（v 1）、規則の一歩も速さ 1（`rule_agent.slide_step`） |
| ランダムに選んだ目的地へ向かう | 合っている | `agent_base.walk`・`reset_target`（goal_margin 1.5 で目的地を引き直す） |
| 配置：エピソードごとに 1/2 で赤と緑を入れ替え、エピソード中は不変 | 合っている（段階I・段階IIのデータ） | `collect_data_v11.py:91-95`（`episode_layouts`）、T2 の test だけ t=50 で入れ替える |
| 規則の一歩：赤の画素の列ごとの数を重みにした円周上の重心の方角、壁で外向き成分を消す、進めないときはランダム | 合っている | `v11core/pleasure.py:87-97`（`color_direction`）、`v11core/rule_agent.py`（`slide_step`、見えない・角ならランダムの一歩＝`a1_source` 2） |
| 確率 p で規則、p = 0 / 0.5 / 0.8 | 合っている | `collect_data_v11.py`（p_rule 既定 0.5）、`collect_data_v13.py`（P_RULE 0.8） |
| 同じエピソード番号で配置・初期位置・休止の時刻が3条件で一致 | 合っている | `collect_data_v11.py` の docstring（エピソードごとに `md5('v11_s1/<split>/<n>')` で再シード、80% も同じ鍵）、`docs/v13_experiment_log.md` §1（休止を含めて一致を確認） |
| 赤から距離5以内のフレーム 約0.03 / 0.33 / 0.85 | 合っている | `docs/v13_results_log.md` §1（0.030 / 0.328 / 0.846） |
| 中央値14ステップで赤の角に着く | 数は合っている（言い方 → A10） | 同 §1 |
| 実際に規則で動いた一歩 80% 57%、50% 48% | 合っている | 同 §1（起きている一歩のうち規則 0.568 / 0.477） |
| 段階I：A-2 はランダムな位置で静止 | 合っている | `collect_data_v11.py`（StayAgent、`self_random_other_stay.yml`） |
| 学習用 3000・評価用 300 | 合っている | `collect_data_v11.py:56`（FULL）、`collect_data_v13.py`、`collect_data_v11_s2t.py:53-54` |
| 段階II：A-1 はランダム、A-2 は自分の視界で同じ規則を確率 0.5 | 合っている | `collect_data_v11_s2t.py:5-7,81-82` |
| 緑好みデータ・好み混合データ（好みはエピソードごとに 1/2） | 合っている（評価の test → A4） | `collect_data_v11_s2t.py:49-54`（`RandomState(md5('v11_s2/pref'))`） |

### 項目 8：快・不快

| 本文の記述 | 判定 | 根拠 |
|---|---|---|
| 範囲 k ＝ 列 8k〜8k+7（カメラの境目） | 合っている | `v11core/pleasure.py` docstring（Bin k = columns 8k..8k+7 = one of the 8 cameras、中心 −157.5° + 45°k） |
| (好きな色 − 嫌いな色の画素数)/128 | 合っている | `pleasure_np`（BIN_PIXELS = 16 × 8 = 128）、モデル設定 `pleasure.scale: 1.0` |
| マスク前の視界から計算 | 合っている | `model/model_v12.py`（`compute_pleasure(x['self_vision'])`：マスクはエンコーダの出力にかかり、入力の視界にはかからない） |
| 好きな色は赤、嫌いな色は緑 | 合っている | `config/model/SuperpositionNetworkMotionPleasureHold/default.yml`（liked Red、disliked Green） |
| 範囲 2〜5 は常に 0 | 合っている | `docs/v11_results_log.md` 583 行（P_A2 の範囲 2〜5 の分散 0.000）、`docs/v11_experiment_log.md` 1298 行、v14 ログ §2.2（|P| の平均が範囲 2〜5 で 0） |

### 項目 9：運動の隠し方

| 本文の記述 | 判定 | 根拠 |
|---|---|---|
| 5〜15 ステップの見える区間と隠れた区間を交互（見える区間から、t=0 は見せる） | 合っている | `model/model_v12.py`（`make_schedule`）、モデル設定 `motion_mask`（len_min 5、len_max 15、n_steps 100） |
| t ≥ 1 の約48% | 合っている | `docs/v12_experiment_log.md` 34 行（0.479）・63 行（0.483） |
| 隠れた区間は m=0、b=1、P を区間の最初の一歩の値に保つ | 合っている | `SuperpositionNetworkMotionPleasureHold.process1_input`（隠れ始めの一歩の視界から計算した P を区間の間保つ） |
| x¹ は 11 次元 [m, P, b] | 合っている | 同、`IN_DIM = 2 + 8 + 1` |
| 段階Iの process-2 は 11 次元すべて零 | 合っている | `forward` の `p2 = torch.zeros_like(p1)`（印も 0） |

### 項目 10：基準エンコーダの値 → **A2**（R² の 2 つの値）

| 本文の記述 | 判定 | 根拠 |
|---|---|---|
| 文献の学習設定（運動入力、入れ替えなし、1000 エピソード）で一から再現学習した重み | 合っている | `exp1_l1_1000/0` ep200（`exp1_mse_1000` → `exp1_l1_1000`、データ `self_random_other_stay_1000`、train 1000） |
| h¹→A-1 0.977、h²→A-2 0.973 | **違う** | `exp1_l1_1000` は 0.9667 / 0.9932（A2） |
| 図5に当たる取り出しのマージン +0.091 | 合っている | `docs/v11_summary.md` 表 2（v4：+0.091、Fig.5 の Decoder `fig5_v4_exp1l1000`）、`docs/v12_summary.md` §9.1（0.799、+0.091）。v12 本番の凍結モデルで v12 側の Decoder を学習し直した確認は +0.075（`docs/v12_summary.md` 表 1 の G4） |
| 自他一致度 0.80 | 合っている | `docs/v12_results_log.md` 28 行（0.799）、`docs/v11_summary.md` 表 2 |
| 一から学習すると 0.18〜0.27 | 合っている | `docs/v12_summary.md` §1（v12 本体の試し学習 0.18〜0.27）・§9.1（E2 の ep200 0.243、試し学習 v12 本体 0.270）、`docs/v12_results_log.md` 28 行（0.183・0.270） |
| 基準エンコーダはすべての条件で同じ、凍結 | 合っている | 全条件の mse 設定の `pretrain_encoder_from` と freeze、CPU で重みの一致を確認 |

### 項目 11：非共有条件

| 本文の記述 | 判定 | 根拠 |
|---|---|---|
| 共有モジュールの LSTM を process-1 用と process-2 用に分ける | 合っている | `model/model_v13.py`（`SplitSuperpositionModule`：`lstm` と `lstm_other`）。NS の mse ep200 の state_dict に `superposition_module.lstm_other.*` がある |
| 層正規化と特徴予測器は分けない | 合っている | `model/model_v13.py` docstring（share_lns（凍結）と特徴予測器は共有）。統合層・復号部も一つのまま（分けていない） |
| process-2 用は process-1 用の写しから始める | 合っている | `copy.deepcopy(shared.lstm)`（乱数を引かないので、初期値・バッチの順・隠す区間・dropout が同じシードの共有ありと同じ） |
| 経験80%のデータで学習 | 合っている | `config/exp/v13ns_s1_mixed_mse.yml`（`v13_s1_mixed80`）。非共有・経験なし（`v13ns_s1_random_*`）もある |
| process-2 の快・不快を受ける入力の重みが一度も学習されない | 合っている（補足） | 入力が常に 0 なので勾配が 0、重みの減衰（0.01）だけがかかる（`docs/v13_experiment_log.md` §19・§21、`docs/v14_summary.md` §2 (3)） |
| L1 期間の凍結 | — | `v13ns_s1_mixed_l1_s*.yml` の freeze `superposition_module` は `lstm_other` も含む（名前の部分一致） |

### 項目 12：段階II

| 本文の記述 | 判定 | 根拠 |
|---|---|---|
| 他者用モジュール：マスク前の視覚エンコーダ2の出力（64 次元）→ LSTMCell（128）→ 全結合 → tanh → 8 次元 | 合っている | `model/modules.py:189-216`（`MotionGeneratorModule`）、`model/model_v12.py:154-158`（マスクの前の `ov_enc` を G に入れる）、`config/model/SuperpositionNetworkMotionPleasureHoldOther/default.yml`（input 64、hidden 128、output 8） |
| $x^2_t = [0, Q2_t, 1]$ | 合っている | `model_v12.py` の forward（`cat[zeros(2), q2, ones(1)]`） |
| G 以外はすべて凍結 | 合っている | `config/exp/v12_s2_*.yml`（freeze：superposition・vision_encoder・vision_decoder・integration・share_lns・feature_prediction） |
| 視覚は t>0 で process-1・2 とも隠す | 合っている | `exp/runner.py:151-152`（`MotionGenerationPredictionRunner`：t>0 で p=1） |
| process-1 には運動を隠さずに入れる | 合っている | モデル設定 `motion_mask.enabled: false`（印 0、今の快・不快） |
| 損失：A-1 の次の視界の絶対誤差と特徴予測の誤差 | 合っている（勾配の道 → A6） | `config/exp/v12_s2_*.yml`（l1_loss）、`exp/runner.py:181-197` |
| AdamW（1e-3、0.01、バッチ 10）、200 エポック | 合っている | `config/exp/v12_s2_pref_mixed_s0.yml` ほか（bias は減衰 0）。v13・2段目 b・c・1ビットの設定はモデルのクラス・pretrain・データだけが違う（`diff` で確認） |
| (ii) 形制限：範囲 2〜5 は 0、0・1 と 6・7 は逆の符号、符号を一つ、範囲ごとの強さはモジュール | 合っている | `model_v12.py` `SuperpositionNetworkMotionPleasureHoldOtherShaped`（signed_pairs：Q2 = [−u m0, −u m1, 0,0,0,0, u m6, u m7]、u = z[2]、m_k = (z_k+1)/2） |
| (iii) 他者遮蔽学習：入力の視界から A-2 を 10〜30 ステップの区間で消す、予測の正解は A-2 が写った視界 | 合っている | `analyze/v12/make_hide_a2_v12.py`（既定 lmin 10、lmax 30、交互、約 50%。A-2 を場の外 (1e4, 1e4) に置いて描き直す）、設定 `v12_s2c_*`（データ `v12_hide_v11_s2_*`、`input_vision: self_vision_in`）、`docs/v12_summary.md` §6（約 47%）。入力の視界は 2 つのエンコーダと G の全部に入る |
| (iv) 1ビット：u ∈ (−1, 1)、Q2 = u c+、c+ = (−0.104, −0.127, 0, 0, 0, 0, +0.127, +0.104) | 合っている（作り方の言い方 → A8） | `model/model_v14.py:23-54`、`onebit.yml` |
| (ii)・(iii) は (i) の結果を見た後に設計した探索 | 合っている | `docs/v12_summary.md` §3・§5・§6（c は「新しい仮説の事前登録」だが正式の主ではない） |

### 項目 13：評価の手順

| 本文の記述 | 判定 | 根拠 |
|---|---|---|
| G1：運動が見えるステップで h¹ から A-1 の位置を線形回帰、0.80 以上 | 合っている | `analyze/v12/eval_trial_v12.py` docstring 1（Ridge(1)、エピソード単位の GroupKFold(5)） |
| G3：共通テスト、t=20 から15ステップ、運動と視覚を隠し（b=1）、快・不快を t=20 の値に保つ | 合っている | `eval_trial_v12.py:137-156`（t<20 は段階Iの流し方、その後 K=15 で `hidden_override=1`、視覚 p=1、P は t=20 の視界から） |
| d = 0〜10、最もよく合う d（L1）、k に対する傾き、g = s赤 − (s緑+s青+sシアン)/3 | 合っている | `analyze/v11/localise_v11.py:71-99`（`dstar`・`slopes`・`g_of`）、`cf_dist_views.h5`（`render_cfdist_v11.py`、場の外の d は無効） |
| G5b：共通テスト、t=20〜29 の10ステップ、a = 0.28、10ステップ後のずれ、(+a) − (−a) を範囲の方角に射影して8範囲の平均 | 合っている（流し方 → A5、読み取り → A9） | `analyze/v12/eval_g5_v12.py:34`（T0 20、A 0.28）、`:130-179`（K=10、最後の一歩の `est − ctrl`、`BIN_UNIT` に射影、範囲の nanmean） |
| 符号に従う分：t<20 は本当の P_A2、t=20 から15ステップ P_A2(20) に保つ、好きな色の方向とほか3色の傾きの差、(止める − 反転)/2 | 合っている | `eval_g5_v12.py:68-126`（`g5a_est`・`g5a_metric`：方向は t=20 の本当の位置から各色の基準点へ）、`analyze/v13/g5a_decomp_v13.py`（反転）、`analyze/v13/across_pairs_v13.py:42-44`・`robust6_v13.py:84-86`（/2） |
| T3：process-2 に [0, P_A2, 1]、他者側の特徴予測の誤差を零と比べる、段階Iだけで決まる | 合っている | `analyze/v11/t1t3_s2_v11.py`（零 = [0,0,0,1]、特徴予測(h²) と次の視界のエンコーダ2の出力の MSE）、項目 6 |
| T2：t=50、緑好き、入れ替えの有無の差 | 合っている | `analyze/v11/eval_s2_v11.py:191-200`（newG = 入れ替え後の緑の範囲、oldG = 入れ替え前）、`collect_data_v11_s2t.py`（`v11_s2t_green_swap` は t=50 の描画の前に入れ替え。t≤49 の視界と A-2 の判断は同じ） |
| T5：同じ配置、A-2 1.0 以内、A-1 3.0 以内、t ≥ 20、1対1 | 合っている | `eval_s2_v11.py:91-`（`t5_pairs(r2=1.0, r1=3.0, tmin=20)`、近い順に貪欲に1対1。対の数が配置ごとに 500 未満なら 1.5 に広げる規則があるが、保存済みの評価はすべて 1.0：対 3305・3018、他者遮蔽学習の隠した test は 1555・1419） |
| T2・T5 の真値の目安 +0.60・+0.42 | 合っている | `docs/v13_results_log.md` §8（+0.604、+0.421） |
| 実験4：好み混合の test 300、[0, c, 1] を全ステップ、T=50、BA（配置×好みの4セル）、正解の決め方 | 合っている | `docs/v14_experiment_log.md` §3 B〜D、`analyze/v14/aggregate_inverse_v14.py` |
| 自己視界基準は L1 期間の最終5点、他者特徴基準は MSE 期間の最終5点 | 合っている | `docs/v14_experiment_log.md` §20・§21 B（項目 2 のとおり L1 期間は復号部だけなので FP は変わらない） |
| 方角の正解（t=1〜50 の変位）、全体の一致 0.707、食い違う 88 エピソード | 合っている | `docs/v14_experiment_log.md` §18・§19 |
| 好み追従率の条件を経験80%の結果を見る前に事前登録、経験50%の結果を見た後に作った | 合っている | `docs/v14_experiment_log.md` §21（v13・v14-C の値を見る前、v12 の §19 の後） |
| 判定の規則（5点中3点、3組中2組、4組で3組） | 一部違う → **A3** | |
| 同じチェックポイントを2回流してビット一致 | 合っている | ep200 を2回（`docs/v13_experiment_log.md` §22・§26・§32、v14 ログ §3 F） |

### 項目 14：組をまたいだ統計

| 本文の記述 | 判定 | 根拠 |
|---|---|---|
| 各組の最終5点の平均を1つの値 | 合っている | `docs/v13_summary.md` §11（late-5 の各点のエピソード平均の平均）、`analyze/v13/across_pairs_v13.py` |
| 組どうしを対にした差の平均と t 分布の 95% CI（6組で自由度5） | 合っている | 同（自由度 5、本番 3 組は 2）。符号検定も出しているが本文では使っていない |
| 事前登録していない記述的な分析 | 合っている | `docs/v13_experiment_log.md` §33 |

---

## C. 確かめられなかった点
- 文献（Noguchi ら）の本文で「Visual Predictor」が統合層（h¹・h² をまとめる全結合）を含む呼び名かどうか（論文の本文は読んでいない）。元のコードでは統合層と復号部が別モジュールで、L1 期間は統合層も凍結する（A1）。
- T3 の D6 と、段階Iの評価の「D6 に当たる副」（`main/d6`）の値がわずかに違う理由（A の補足）。
- 「四隅の外側」の立方体の位置（(±13, ±13)）はプロジェクトのメモの記録で、今回コード（`sim_v10/layout.py`・field.obj）では確かめていない。
