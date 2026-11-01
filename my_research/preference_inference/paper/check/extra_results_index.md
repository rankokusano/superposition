# 追加実験（作業記録 §35〜§39）の結果の一覧と、記録の残り方

- 作成：2026-10-31（第13弾の後）。既存のファイルは変えず、確認と一覧づくりだけをした。
- 追記：2026-11-01（第14弾の段階Cの行と報告のファイルを足した）。
- 追記：2026-11-01（第15弾 D1・D2 の行と報告のファイルを足した）。2026-11-02（第16弾の行と報告のファイルを足した）。2026-11-02（第17弾の行と報告のファイルを足した）。
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
| この一覧 | `paper/check/extra_results_index.md` | この一覧と同じコミット |
| コード（計算・集計・図・実行） | `analyze/paper_extra/e1_*.py`・`e2_*.py`・`e2pos_*.py`・`c_run.py`・`train_paper_e1.py`・`run_e1_*.sh`・`run_e2*.sh`・`vdir_*.py`・`vdirb_*.py`・`run_vdir_c.sh`・`run_vdirb_eval.sh`、`simulation/make_signrand_paper_e1.py`・`collect_data_vdir_b.py` | 各コミット（`git log -- <file>`） |
| 設定 | `config/exp/paper_extra/E1_signrand50/*.yml`（7 ファイル） | `a1770ba` |

- **git に入っていないもの（慣例で、入れていない）**：
  - 指示書：`paper/check/paper_instructions_round11.md`、`docs/paper_instructions_round12.md`・`round13.md`・`round13_rev2.md`。
    - 中身の要点は作業記録の各節に写してある。
  - 図：PDF・PNG（下の §3）。

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

- 学習に使ったほかのチェックポイント（経験なし・経験50%・経験80%・符号無作為（経験80%）・好み切替）は、§35 より前の作業（v12〜v14、第6弾）のもので、それぞれの作業記録にある。
- 作り直しの手順は、どれも作業記録の該当の節に、コード・鍵・コマンドとして書いてある。
  - 例外：E1 の学習の GPU での再現性は確かめていない。
  - 例外：指示書の原本は git に入っていない。
