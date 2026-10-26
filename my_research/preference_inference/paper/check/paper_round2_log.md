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

## 2. 表・図との一致（指示書 §2.4。こちらで確認）
- 本文が読む図 7 枚（env_conditions、network、central、projection_error、a_cross_delta、inverse、write_vs_choose）と表 8 つ（conditions、metrics、group_values、projection、inverse、alignment、module、gates）は、すべて `paper/figs/`・`paper/tables/` にある。`draft/` の中に図表の写しはない。
- 図の説明（\caption）と図：env_conditions（(a)〜(e)）、network（(a)(b)、差し込み、斜線）、central（線の種類・印・白い菱形・事前登録の区別）、projection_error（(a)〜(d)）、inverse（(a)〜(c)）、write_vs_choose（(a)(b)、塗り・白抜きの菱形）は、図と合っている（PDF を画像にして目で確かめた）。
- **a_cross_delta**（追加実験 A の図、目で確かめた）：図は (a) 緑好みデータ・(b) 好み混合データの2パネルで、太い線が組1〜6 の平均、細い線が組ごと（実線＝組1〜3、点線＝組4〜6）、縦軸は「改善 Δ（真値 − 零）」。本文の説明（results.tex 186 行）は縦軸・横軸だけで、(a)(b) と線の意味が書かれていない（説明の不足。食い違いではない）。
- 本文の図のパネルの参照（methods 30・46、results 45・70・85・102・135・144・164・238・262・283、discussion 85）は、すべて図にあるパネルを指している。
- 小さな言い方の違い：図3 の (a) の題は「自己側の積み上がり」、本文の説明は「自分側の積み上がり」。

## 3. 照合の結果のまとめ（2026-10-26）
- 節ごとの照合（5つ）を `analyze/paper/check_round2/part_*.md` に書き、`paper/check/numbers_check.md` にまとめた（先頭に違う・範囲の取り方が違う・見つからないの表 32 項目、続いて §2.2〜§2.5、節ごとの全項目の表）。
- 主な指摘のうち 2 つはこちらでも保存値で確かめた：(1) 他者特徴基準（MSE 期間の最終5点）での経験80% の BA は 6 組とも 0.715〜0.801 で、方角だけの基準 0.708 を点推定で上回る（`aggregate_v13mse_s0to5.json`）。(2) 好み混合データでは組3 も真値 0.4141 が零 0.4127 より悪く、零より悪いのは 3 組（緑好みでは 2 組）。
- 再計算が要るもの：付録 188 行の「5〜8%」「80〜90%」を本文の条件のモデルで確かめること（計算していない）。モデルと方角だけの基準の対にした差の CI（計算していない）。
