# 論文の表（第1弾、2026-10-26）

- 置き場所：Overleaf の `tables/` にそのまま置く（本文の `\input{tables/<名前>}`）。中身は `tabular` だけ（`table` 環境と `\caption` は本文側）。`booktabs` を使う。
- 作り直し：コンテナの中で `cd /work/.worktrees/v11/my_research/preference_inference/analyze/paper && python tables_paper.py`（8 つを一度に書く）。
- 文字：書き出すときに `iso2022_jp` で符号化できることを確かめている（JIS X 0208 の外の文字があると止まる。「—」は `---` にした）。
- 組み方の確認：jsarticle（upLaTeX、11pt、A4）で `\small` をかけた `table` 環境に入れ、8 つとも本文幅に収まりエラーがないことを確かめた。**本文側の `table` 環境で `\small` をかける前提**。`inverse` と `module` はファイルの頭で `\tabcolsep` を 3.5pt にしている（その表の中だけに効く）。
- 数字の桁：出どころの JSON の値を 3 桁で書いた（数値ログと同じ桁）。md から取った値は md のとおりの桁。範囲は「〜」、区間は [下限, 上限]。

| ファイル | 中身 | 出どころ |
|---|---|---|
| `conditions` | 表1：条件と組（経験なし・50%・80%・非共有の p、共有の有無、段階I・実験1・2 の組、実験4 の組）。下段に他者用モジュール4通り（出力の形、学習データの視界、段階Iの条件と組、扱い）。好み切替条件は注 | `paper/notes/論文_名前の対応表.md` §2・§3・§5。モジュール名と組は `paper/check/methods_check.md` の項目4〜5（自由出力 `v12_s2_*`・`v13_s2_*_mixed` 組1〜4、形制限 `v12_s2b_*` 組1〜4、他者遮蔽学習 `v12_s2c_*` 組1〜3、1ビット `v14_s2u_pref_*` 組1〜3）。扱いは v12_summary §3、v13_summary §4、v14_summary §6 |
| `metrics` | 表2：指標の定義と事前登録の区別（14 行） | 対応表 §8、v13_summary §3.1・§10・§11、v14_summary §6、v14 ログ §21。評価データの区別は methods_check A4 |
| `group_values` | 表3：G3・符号に従う分（緑好み・好み混合）・G5b の組ごとの値（最終5点の平均、経験なし / 50% / 80%）、非共有（組1〜3）、組をまたいだ差（6組、t の 95% CI） | `data/result/v13/extra/across_pairs.json`（`per_pair`・`tests` の `all6`。v13 数値ログ §11 と同じ値）、非共有は `paper/figs/central_values.json`（v13 数値ログ §12 と同じ値） |
| `projection` | 表4：真値条件の予測誤差（他者側の特徴予測、零・真値・生成）を経験なし・50%・80% × 組1〜4 × 2 データで。下段に一歩の種類ごと（組1〜3 の範囲） | `paper/figs/projection_values.json`（元は `data/result/{v12,v13}/stage2/t1t3/`）、`data/result/v13/post/d6_split/summary.json`。生成は G の最終5点の範囲、零と真値は G によらない1つの値 |
| `inverse` | 表5：(a) 自己視界基準の BA（T = 50、L1 期間の最終5点の範囲と合格の点）を経験80%・50%・経験なしで組1〜6、方角だけの基準の BA（全体と規則の一歩が少ない3分の1）。(b) 経験80% の他者特徴基準の合格の点、好み追従率（他者特徴基準は MSE 期間、自己視界基準は L1 期間の最終5点の範囲と CI の下限 > 0.5 の点、他者特徴基準には書き方）、(c) 経験80% − 非共有（組1〜3、合格の点と ep200 の差） | `data/result/v14/inverse/aggregate_v13_s0to5.json`・`aggregate_v12_s0to5.json`・`aggregate_v13mse_s0to5.json`・`aggregate_v13_vs_ns.json`、`paper/check/follow_pref_late5.json`、`paper/check/direction_baseline.json`。v14_results_log §9・§7 の値と一致（`inverse_values.json` に表の値を保存） |
| `gates`（付録） | 表A1：自己位置の追跡・価値と行動・色の結びつき・自己側の積み上がり・G5a・G5b の点（5点中）を、経験50%・80%（組1〜6）、非共有（組1〜3） | `data/result/v13/extra/robust6.json`（v13_summary §10 と同じ）。G5b は「経験あり − 経験なし」の項目 (c)（関門の判定と同じ基準）。非共有は v13_summary §5 の表5：予測の偏り M1(c)・反事実の視界への寄り M1(d)・G3・G5a（①②とも）・G5b は組の差が仮説の向きで CI が 0 を含まない点（各組 5/5・5/5・5/5・0/5・0/5）、色の基準は値の範囲、自己位置の追跡は値 0.739〜0.762（5 点とも 0.80 未満） |
| `module`（付録） | 表A2：他者用モジュール4通り × 条件の (a) T2・T5 の合格の組、合格の形の組の経験ありの値、Q2 と本当の快・不快の相関、範囲2〜5 の分散の割合、(b) T3（生成・真値・零）と好み追従率 | (a) は md：自由出力 50% v12_summary §4（相関は v13 ログ §22 の v12 正式 ①）、80% v13_summary §4・v13 ログ §22・§26（相関は ① の4組）、形制限 v12_summary §5、他者遮蔽 v12_summary §6、1ビット 50% v14_results_log §3。1ビット 80% は `data/result/v14/stage2u/judge_b_extra_v13.json`（T5、条件 (a)）・`eval/pref_mixed80_s*_ep200.json`（相関）。(b) の T3 は好み混合データでそろえて保存値から：自由出力 `stage2/t1t3`、形制限 `data/result/v12/stage2b/t1t3`、1ビット `stage2u/t1t3`（ep200）。他者遮蔽学習の T3 だけ v12_summary §6 の値（隠れているフレームの自己の視界の誤差、指標が違う）。好み追従率は `paper/figs/write_vs_choose_values.json`（段階IIの最終5点、組1〜3） |
| `alignment`（付録） | 表A3：自他一致度を、視覚エンコーダの扱いと入力・環境・データで並べた | v11_summary §2.4 の表（基準エンコーダ 0.80、基準エンコーダを凍結・入れ替えあり 0.73（v10 B3-frozen）、価値の信号で学習し直し 0.25（v5_base_l1）、基準から微調整 0.26〜0.32（v10 B3-ft）、快・不快だけで一から 0.16〜0.30（v11））、v12 ログ §8（試し学習 (P-a) 0.47〜0.48、v12 本体 0.18〜0.27）、v12_summary §9.1（基準 0.799・マージン +0.091、E1 0.740（ep100 0.783）・+0.075、E2 0.243・+0.026） |
