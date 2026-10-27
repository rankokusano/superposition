# 論文 第3弾の作業記録（短縮版の表・図・数字の照合、査読の指摘への再分析）

- 指示書：`paper/check/paper_instructions_round3.md`（草野さんが置いたもの）
- 原稿：`paper/draft_v2/`（main_v2.tex、supplement_v2.tex、sections_v2/、supp_v2/、tables/v2_design.tex（仮））。読むだけで、直さない。
- 優先順位（草野さん、2026-10-27）：
  1. 零の条件の結果の確定と報告。
  2. 作業4 の B1〜B3（数字だけ）。
  3. 作業3（数字の照合）。
  4. 作業1・5（表と図）。
  - B4 は計算量の見積もりだけを報告し、了承まで実行しない。
- GPU は使わない。GPU0 は使わない。`git add -A` は使わない。

## 0. 零の条件（快・不快を常に零）の確定（2026-10-27）
- 学習・評価・判定は済んでいた（`docs/paper_extra_log.md` §28、コミット `a39b3d4`。3組とも ep200 まで NaN なし、評価の ep200 の2回目は bit 一致）。
- 一次の集計 `data/result/paper_extra/C_signrand/summary.json` の `pzero` を読み直し、§28.2 の表の値と一致することを確かめた。
- `paper/figs/c_signrand.pdf` は零の条件を入れた版に差し替え済み（§28.5）。
- 本文 4.3節（results.tex 87 行）と限界（discussion.tex 68 行）の【結果待ち】に入る数字は、報告に書く。
- 実行中のジョブはない（GPU1〜7 は空き。GPU3 に 2 GB があるが、このチャットのものではない）。
