# 論文の図（第1弾、2026-10-26）

- 置き場所：Overleaf の `figs/` にそのまま置く（本文の `\includegraphics{figs/<名前>}` と同じ名前）。PDF がベクタの本体、PNG は確認用。
- 作り直し：コンテナ `kusano_research` の中で `cd /work/.worktrees/v11/my_research/preference_inference/analyze/paper && python <スクリプト>`。共通の書式は `analyze/paper/paper_style.py`。
- **日本語のフォント**：コンテナに日本語のフォントがないので、作り直す前にホストから写す（リポジトリには置かない）：
  ```
  docker exec kusano_research mkdir -p /tmp/paperfonts
  docker cp /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc kusano_research:/tmp/paperfonts/
  ```
- **フォントの埋め込み**：matplotlib 3.4 は TrueType（Type 42）を部分埋め込みしないため、Noto の全体が入って PDF が 16 MB になった。ghostscript での部分化も .ttc を読めず別のフォントに置き換わったので使わない。**Type 3 で埋め込む**（使った字形だけ、PDF は 50〜70 KB）。図の中の文字は PDF 上で検索・コピーできない。投稿先が Type 3 を受け付けない場合は、matplotlib を 3.6 以降にして Type 42 にする（要相談）。
- 色：条件ごとに固定（経験なし 灰・丸、経験50% 橙・四角、経験80% 青・三角、非共有 緑・白抜きの菱形）。色だけに頼らないよう記号の形も変えた。
- 目視：6 枚とも PDF を画像にして確かめた（2026-10-26）。

| ファイル | 中身 | 元の値 | スクリプト | 目視 |
|---|---|---|---|---|
| `network` | 図1。段階I（上）と段階II（下）。視覚エンコーダ1・2 は凍結（斜線）。共有モジュール（LSTM 128）の process-1・2、process-1 の入力 [運動 2, 快・不快 8, 隠れフラグ 1]、process-2 は段階Iで零・段階IIで [0, Q2, 1]。視覚予測器は「統合層（$h^1, h^2$ をまとめる全結合 256→64）」と「復号部」に分け、L1 期間は復号部だけを学習することを書いた（`paper/check/methods_check.md` A1）。他者用モジュール G はマスク前の視覚エンコーダ2の出力を読む。差し込みで非共有条件（LSTM を process-1 用と process-2 用に分ける）。 | `docs/figs/network_v11_v12.svg`（v12 の部分）とコード（`model/model_v12.py`、`config/exp/v12_s1_*_l1_*.yml`） | `fig_network_paper.py` | 済み |
| `env_conditions` | 図2。(a) アリーナ（±10）と四隅の外の立方体（中心 (±13, ±13)、`sim_v10/layout.py`。立方体の大きさは模式）、元の配置と赤・緑の入れ替え。(b) A-1 が (−8, 8) にいるときの全方位の視界（配列をそのまま RGB で表示。空が茶色に見えるのは元の図と同じ）と8範囲の快・不快（元の配置・入れ替え）。(c) 段階Iの学習データ（train 3000）の A-1 の滞在位置、経験なし・50%・80% × 2 配置（色の上限 0.02） | `data/result/v11/gate61/gate61_pleasure_views.npz`（`analyze/v11/gate61_v11.py`）、学習データ `data/data/{v11_s1_random,v11_s1_mixed,v13_s1_mixed80}/data.h5`（`analyze/v13/gate61_v13.py` と同じ集計） | `fig_env_paper.py` | 済み |
| `central` | 図3（中心の図）。(a) 自己側の積み上がり g、(b) 符号に従う分（緑好みデータ）、(c) 符号に従う分（好み混合データ）、(d) 他者側の符号応答 G5b。各組の最終5点の平均（組1〜3 灰の実線、組4〜6 橙の破線）、6組の平均と t の 95% 信頼区間（自由度5）、非共有条件（経験80%、組1〜3）を白い菱形 | `data/result/v13/extra/across_pairs.json`（`analyze/v13/across_pairs_v13.py`）、非共有は `analyze/v13/fig_central_v13.ns_values`（NS の 1 エピソードごとの値から同じ平均の取り方）。描いた値は `central_values.json` | `fig_central_paper.py` | 済み |
| `projection_error` | 図4（投影の誤り）。(a) 他者側の特徴予測の誤差（T3）を、process-2 に零・真値（本当の快・不快）・他者用モジュール（自由出力）の生成を入れた場合で、経験なし・50%・80%、組1〜4（組4 は橙の破線）、緑好み・好み混合データ。零と真値は段階Iだけで決まる（ep160〜200 の G で同じ値、確かめた）。生成は G の最終5点の平均（棒は範囲）。(b) 真値条件の誤差を A-2 の一歩の種類（好みに従った一歩・従わなかった一歩・休止）で分けたもの（組1〜3。ログにある集計） | `data/result/{v12,v13}/stage2/t1t3/<set>_<cond>_s<s>_ep<e>.json`（v13 ログ §24 の T3）、`data/result/v13/post/d6_split/summary.json`（v13 ログ §25）。経験なしは v12 の random（v13 のファイルとも一致を確かめた）。描いた値は `projection_values.json` | `fig_projection_paper.py` | 済み |
| `write_vs_choose` | 図5。好みと方角が食い違う 88 エピソードで好みに従う割合。当てて選ぶ（実験4：自己視界基準は L1 期間、他者特徴基準は MSE 期間）と書き出す（1ビットモジュール、段階II）を、経験50%・経験80%、組1〜3。点は最終5点の平均、棒は範囲。0.5 に破線 | `paper/check/follow_pref_late5.json`（`paper_round1_log.md` §3・§4。保存済みの v14-A の出力を同じ関数で数え直したもの）、`data/result/v14/stage2u/u_direction{,_v13}.json`。元の図 `data/result/v14/summary_figs/fig3_write_vs_choose.png`（下段）は ep200 の1点と CI。描いた値は `write_vs_choose_values.json` | `fig_write_vs_choose_paper.py` | 済み |
| `inverse` | 図6（実験4）。(a) 自己視界基準の正解率（BA）と照合の長さ T、組1〜6 の各パネルに経験80%・経験50%・経験なし・非共有（組1〜3）。点は L1 期間の最終5点の平均、棒は範囲。**方角だけの基準の BA**（黒の破線、灰の帯は 95% 信頼区間）を同じ軸に描いた。(b) 好み追従率（他者特徴基準は MSE 期間の最終5点、自己視界基準は L1 期間の最終5点）、経験80% の上の数字は CI の下限 > 0.5 の点の数（5点中） | `data/result/v14/inverse/aggregate_v13_s0to5.json`・`aggregate_v12_s0to5.json`・`aggregate_v13_vs_ns.json`（元の図 `fig1_ba_vs_T.png` と同じ）、`paper/check/direction_baseline.json`、`paper/check/follow_pref_late5.json`。経験50% の組4 の MSE 期間は保存がないので描いていない | `fig_inverse_paper.py` | 済み |
