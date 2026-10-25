# 論文 第1弾の作業記録（方法の節の確認、図1〜6、表1〜5・A1〜A3）

- 指示書：`docs/paper_instructions_round1.md`（2026-10-26、論文の執筆のチャット）
- 作業場所：worktree `.worktrees/v11`、ブランチ `feature/v11-self-experience`。出力は `paper/`・`analyze/paper/` だけ。v11〜v14 のコード・設定・データ・結果は読むだけ。
- 新しい学習はしない。GPU は使わない（使う必要が出たら起動の前に確認を取る）。

---

## 0. 読むものの用意（2026-10-26）
- 指示書 §0 の3つの文書（`論文_執筆メモ.md`・`論文_名前の対応表.md`・`methods.tex`）はサーバーのどこにもなかった（`find /` で確認）。チャットに貼られた内容をそのまま `paper/notes/` に保存した。指示書そのものも `docs/paper_instructions_round1.md` に保存した。
- 執筆メモ・対応表が参照する `claude/v13設計メモ.md` などの `claude/` 以下の文書はリポジトリにない（執筆のチャットのプロジェクトにある）。数字の出どころはリポジトリの `docs/*_summary.md`・`*_results_log.md`・`*_experiment_log.md` と保存済みの結果だけを使う。
- 日本語のフォント：コンテナ `kusano_research` には日本語のフォントがない（Droid Sans Fallback だけ）。ホストの `NotoSansCJK-Regular.ttc`・`NotoSansCJK-Bold.ttc`・`ipag.ttf` をコンテナの `/tmp/paperfonts/` に写した（リポジトリには置かない）。matplotlib 3.4.3 で Noto Sans CJK JP を使った PDF（Type 42）が書けることを確かめた。図のスクリプトは `/tmp/paperfonts/` のフォントを読み込む（コンテナを作り直したら写し直す。手順は `paper/figs/README.md`）。

## 1. 測り方の記録：方角だけの基準の BA（指示書 §2.1 の 3。**計算の前に記録**。データだけで決まる量、CPU）
- **位置づけ**：事前登録外の記述（論文の表5・図6 に並べる比較の基準）。v14 のどの判定も変えない。
- **データ**：好み混合データの評価用 300 エピソード（`v11_s2_pref` の test。`analyze/v11/eval_s2_v11.load_set('pref')` で読む）。
- **方角の選択**（v14 ログ §18 と同じ定義）：A-2 の t = 1〜T の変位 Δ = `other_position[T+1] − other_position[1]` を、範囲 0・1 の方向と範囲 6・7 の方向（`v11core.pleasure.BIN_UNIT` の2範囲の平均を単位ベクトルにしたもの）に射影し、範囲 6・7 の方向の射影が大きければ c+、そうでなければ c−。等しい場合も c−（§18 の実装 `np.where(disp @ u67 > disp @ u01, 1, -1)` と同じ）。
  - T = 50 は §18 と同じ（`other_position[51] − other_position[1]`）。
  - T = 5, 10, 20, 30 は同じ形で `other_position[T+1] − other_position[1]`。
  - **T = 100 は位置が t = 100 までしかない（101 フレーム）ので `other_position[100] − other_position[1]`（t = 1〜99 の 99 歩）とする**。
- **正解**：好みの正解（v14 ログ §3 C の `correct_sign`：A-2 の好きな色が範囲 6・7 にあれば c+、範囲 0・1 にあれば c−）。v14-A の出力（`data/result/v14/inverse/v12_mixed_0_ep200_pref.npz`）の `correct_sign`・`layout`・`a2_liked` を使う（どのモデルの出力でもデータの値は同じ。全モデルで一致することも確かめる）。
- **BA**：配置 × 好みの4セル（`layout*2 + a2_liked`）の正解率の平均（`aggregate_inverse_v14.ba` と同じ）。
- **95% CI**：エピソードのブートストラップ 1000 回（`aggregate_inverse_v14.boot`：`RandomState(0)`、4セルがすべて入る標本だけ）。
- **素の一致率**（BA ではない、エピソードの単純平均）も出し、T = 50 で v14 ログ §19 の 0.707 と一致するかを確かめる。
- **規則の一歩が少ない3分の1**（§18 と同じ：t = 1〜50 の起きている一歩のうち `a2_source` = 1 の割合が 33.3 パーセンタイル以下、117 エピソードのはず）でも、同じ BA・CI・素の一致率を出す（T = 50 で素の一致率 0.607 と一致するかを確かめる）。部分集合のブートストラップも同じ関数（部分集合の中で4セルがすべて入る標本）。
- 実装 `analyze/paper/direction_baseline_paper.py` → `paper/check/direction_baseline.json`。コンテナの中で CPU だけで流す。
