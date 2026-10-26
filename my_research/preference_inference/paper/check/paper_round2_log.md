# 論文 第2弾の作業記録（本文の数字の照合）

- 指示書：`paper/check/paper_instructions_round2.md`（草野さんが置いたもの。これを正とする）
- 対象：`paper/draft/`（読むだけ、変えない）。表は `paper/tables/`、図は `paper/figs/` を正とする。
- GPU は使わない。新しく書くのは `paper/check/paper_round2_log.md`・`paper/check/numbers_check.md`・`analyze/paper/check_round2/` だけ。

## 0. 置かれたファイル（2026-10-26）
- `paper/draft/` の直下に 9 つ（`sections/` の下ではない）：main.tex、intro.tex、related.tex、methods.tex、results.tex、discussion.tex、conclusion.tex、appendix.tex、references.bib。
- 指示書 §0 の「7つの .tex と main.tex・references.bib」はそろっている。`latexmkrc` はない。
- 注意：main.tex は `\InputIfFileExists{sections/intro.tex}` などで読むので、今の置き方（直下）のままでは各節が「未執筆」として組まれる。照合には差し支えない（報告する）。
- 要旨は main.tex の abstract にあるので、照合の対象に含める。
- 追加実験 A の出どころ：`docs/paper_extra_results_log.md`・`docs/paper_extra_log.md`（同じ worktree で別のチャットが B・C を実行中。そのファイル・ジョブには触れない）。

## 1. 進め方
- 節ごとに分けて照合する（要旨・序論・関連研究・おわりに／方法・付録／結果／考察／文献）。各部分の結果は `analyze/paper/check_round2/part_*.md` に書き、`paper/check/numbers_check.md` にまとめる。表・図との一致（指示書 §2.4）と、複数の場所の食い違い（§2.2）はまとめのときに見る。
