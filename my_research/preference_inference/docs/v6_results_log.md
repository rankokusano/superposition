# v6 数値の一次記録

全実験の測定値を時系列で記録する。各エントリは以下をセットで含める（`docs/v6_instructions.md` §11.3）：

- 実験ID（exp_config名 + seed + epoch）
- git commit hash
- 使用シード（学習・評価とも）
- 評価データセットとsplit（`eval`=訓練split か `test`=held-out か）
- late-5平均 ± inter-ckpt sd（単一チェックポイントの値は使わない）
- 収束判定の結果（§4.2の基準、`min(B)`の値も併記）

数値だけを転記せず、判定基準（合格/不合格）も併記すること。詳細な解釈・設計判断は `docs/v6_experiment_log.md` に書く。ここは数値のみ。

---

## S1: 報酬条件付き critic の学習

**試行1（2026-09-17）**：commit `5390c41` + `docs/v6_experiment_log.md`未コミット更新分。seed=0のみ（単一シード、ゲート判定段階のため可）。学習：`simulation/train_rl_v6.py --seed 0`、GPU7、1000episode×100step、所要時間 約62分（15:49開始, 16:51終了。§6.1の見積もり5〜10hより大幅に短い）。評価：`torch.manual_seed`等の固定はまだ実施していない（health check/plot_q_mapは決定論的な`critic.eval()`前提の推論のみで、確率的要素は`actor.sample`の`rsample()`のみ。再現性の厳密な検証は今回未実施、次回以降に追加）。

**critic健全性チェック**（`analyze/check_critic_health_v6.py --critic_path data/model/v6_rl_critic_seed0.pth --camera self --scan_self`）：

| r条件 | action_std | 閾値0.05 | corner_std | 閾値0.05 | argmax(action=(1,0)) | argmin |
|---|---|---|---|---|---|---|
| A1 (+1,-1,0,0) | 0.038874 | **FAIL** | 0.639519 | PASS | Cyan（期待:Red） | Green |
| A2 (-1,+1,0,0) | 0.013363 | **FAIL** | 0.238180 | PASS | Center（期待:Green） | Red |
| unseen (0,+1,-1,0) | 0.024114 | **FAIL** | 0.164618 | PASS | Cyan（期待:Green） | Green（期待:Blue） |

**値域の一致**：corner-std比 A1/A2 = 2.69x（health check）／グリッド全体std比 2.47x（`plot_q_map_v6.py`、20×20グリッド）。**閾値3.0x以内でPASS**（v4の0.63 vs 0.009 ≈ 70xから大幅改善）。

**Q値空間マップ**（20×20グリッド、`analyze/plot_q_map_v6.py --seed 0`）：
- 保存先：`/home/kusano/superposition/my_research/preference_inference/data/result/v6_baseline/q_map_v6_seed0.png`（3条件並列、共有カラースケール）
- 個別：`q_map_v6_seed0_A1_true.png`、`q_map_v6_seed0_A2_true.png`、`q_map_v6_seed0_unseen.png`
- 統計：`q_map_v6_seed0_stats.json`（A1: min=-1.7475 max=1.3535 std=0.5095／A2: min=-2.4016 max=-0.6627 std=0.2066／unseen: min=-2.0167 max=0.2107 std=0.1948）

**総合判定：FAIL**（§3.0のS1判定基準に対して）。詳細は`docs/v6_experiment_log.md` §9。

**試行2（2026-09-19、hindsight relabeling追加、commit `e1f2de6`以降）**：`--tag relabel`、K_RELABEL=10、MEMORY_SIZE=100000、他は試行1と同一（1000ep×100step、seed0、GPU7）。

| r条件 | action_std（閾値0.05） | corner argmax（期待） | corner argmin |
|---|---|---|---|
| A1(+1,-1,0,0) | 0.0068 FAIL | Cyan（期待Red）FAIL | Green |
| A2(-1,+1,0,0) | 0.0368 FAIL | Green（期待Green）PASS | Blue |
| unseen(0,+1,-1,0) | 0.0361 FAIL | Green（期待Green）PASS | Blue（期待Blue）PASS |

値域の一致：corner-std比1.74x、グリッド全体std比1.66x（試行1は2.69x/2.47x）、PASS。

Q値空間マップ：`data/result/v6_baseline/q_map_v6_seed0_relabel.png`ほか（個別・統計json同ディレクトリ）。**全グリッド可視化により、A2・未知rの5点PASSはランドマーク識別ではなくx座標（左右）への単調依存が偶然一致した結果である疑いが強いと判明**（A1はy軸、A2は同じ|r|でx軸に依存——単純な符号反転になっていない）。詳細解釈は`docs/v6_experiment_log.md` §12。

**総合判定：FAIL**（改善傾向はあるが、視覚検証により5点チェックのPASSの信頼性が低いと判断）。次のアクションとしてFiLM条件付けの実装を提案（ユーザ判断待ち）。

**試行3：seed0/1/2 relabel 比較（2026-09-20、判定基準は`docs/v6_experiment_log.md` §14で事前登録）**：seed0=試行2をそのまま流用、seed1・seed2は`--seed {1,2} --tag relabel`で新規学習（GPU7/GPU6並列、各1000ep、K_RELABEL=10、commit `a879e10`時点のコード）。

| seed | A1 action_std | A2 action_std | unseen action_std | state_std（3条件） | 健全か |
|---|---|---|---|---|---|
| 0 | 0.0068 | 0.0368 | 0.0361 | 全PASS | NO |
| 1 | 0.0087 | 0.0107 | 0.0088 | **全て0.0（完全崩壊、Q=定数）** | NO |
| 2 | 0.0110 | 0.0146 | 0.0114 | 全PASS | NO |

seed1のQ値は全条件・全位置で完全に同一値（A1=0.8910、A2=-0.3729、unseen=0.1598）。
seed2：A1はargmax=Cyan（期待Red）、A2はargmax=Green・argmin=Blue、unseenはargmax=Green・argmin=Blue（ただしA2/unseenのグリッドstdは0.093/0.096と極小で、ほぼ一様）。値域の一致（corner-std比）：seed2は1.16x（seed1は両条件0.0のため無意味）。

Q値空間マップ（全グリッド）：`/home/kusano/superposition/my_research/preference_inference/data/result/v6_baseline/q_map_v6_seed{1,2}_relabel.png`（個別・stats.json・grids.npz同ディレクトリ）。

**総合判定：判断保留**（§14.3第3行：3seedとも健全でない）。詳細は`docs/v6_experiment_log.md` §15。

**対照実験（r関連を全て外す、2026-09-20、`train_rl_v6.py --control`、commit `3bdd1c9`のコード、判定基準は`docs/v6_experiment_log.md` §16.4で事前登録）**：seed0/1/2、各1000ep、GPU5/6/7並列。元の`analyze/check_critic_health.py`で測定（生出力`data/result/v6_baseline/control_health_seed{0,1,2}.txt`）。

| seed | action_std | state_std | 判定 |
|---|---|---|---|
| 0 | 0.162258 | 1.078918 | HEALTHY |
| 1 | 0.105187 | 1.368364 | HEALTHY |
| 2 | 0.077260 | 1.181006 | HEALTHY |

（参考：v3 A-1の同seed=0.119/0.202/0.025。rありv6の12値は0.007〜0.039）。最大値0.162 ≥ 0.12 → §16.4の行1（実装に退行なし、r条件付け〔またはリラベリング・buffer変更〕が疑わしい）。詳細と留意点は`docs/v6_experiment_log.md` §16.6-16.8。

**試行4：(i)(ii)並行実施（2026-09-21、コードcommit `3e17de4`、判定基準は`docs/v6_experiment_log.md` §17で事前登録）**：5本とも1000ep（GPU1/2/4/5/6並列）。生出力`data/result/v6_baseline/health_{norelabel,film}_seed*.txt`、全グリッド`q_map_v6_seed{1,2}_norelabel*`・`q_map_v6_seed{0,1,2}_film*`。

(i) rあり・リラベリングなし（seed0=試行1）：

| seed | A1 | A2 | unseen | 平均 |
|---|---|---|---|---|
| 0 | 0.0388 | 0.0134 | 0.0241 | 0.0253 |
| 1 | 0.0199 | 0.0224 | 0.0136 | 0.0186 |
| 2 | 0.0252 | 0.0099 | 0.0172 | 0.0174（state_std=0、完全崩壊） |

→ §17.2 行1（r単独でaction感度が下がる、非定常性が最有力）。

(ii) FiLM（relabel recipe + FiLM）：

| seed | A1 | A2 | unseen | 健全 | 基準A/B/C | D（目視） | 軸2 |
|---|---|---|---|---|---|---|---|
| 0 | 0.2088 | 0.0376 | 0.0324 | NO | NG/NG/NG | NG（ほぼ一様） | NO |
| 1 | 0.0088 | 0.0190 | 0.0105 | NO | OK/OK/OK | OK（unseenは留意付き） | YES |
| 2 | 0.0180 | 0.0127 | 0.0194 | NO | OK/OK/OK | OK（unseenは留意付き） | YES |

軸1（≥2/3が健全）：**FAIL**（0/3）。軸2（≥2/3がA〜D）：**PASS**（2/3）。→ §17.3：**軸1 FAIL × 軸2 PASS**（rへの応答は直ったがaction感度は低いまま）。全グリッド：`/home/kusano/superposition/my_research/preference_inference/data/result/v6_baseline/q_map_v6_seed{0,1,2}_film.png`。詳細・留意点は`docs/v6_experiment_log.md` §18。

**(c) 8方向プローブQの方向情報（2026-09-22、§19で事前登録、`analyze/probe_q_direction_info_v6.py`）**：c1=r=A-1・r3_stay test self_vision、c2=r=A-2・r2_a1random_a2rl test other_vision（各30,300フレーム）。基準=`v3_rl_critic.pth`（config実使用）。出力`data/result/v6_baseline/probe_dirinfo_v6.{txt,json}`。

| 候補 | ρ_c1 | ρ_c2 | スコア(min) |
|---|---|---|---|
| FiLM seed1 | 0.038 | 0.015 | 0.015 |
| FiLM seed2 | 0.279 | 0.049 | 0.049 |

最良0.049 < 1/5 → §19.4の行2（方向情報が失われている）。c1で√D（方向間の標準偏差）：v3基準0.098、対照0.08〜0.10、リラベリングなしr条件付き0.03〜0.06（向きの相関+0.64〜+0.94）、リラベリングありr条件付き（FiLM含む）0.01〜0.03（相関≈0）。詳細・事後観察・限界は`docs/v6_experiment_log.md` §20。

**(a') FiLM＋リラベリングなし（2026-09-28、§21で事前登録、commit `c644c40`）**：seed0/1/2、GPU1/2/3並列。seed1は状態崩壊（4隅+中心のQが全て同一値）で判定から除外。

| seed | ρ_c1 | dirCorr_c1 | ρ_c2 | dirCorr_c2 | action_std健全 | 基準A/B/C |
|---|---|---|---|---|---|---|
| 0 | 0.424 | +0.658 | 0.936 | **-0.527** | NO | NG |
| 2 | 0.734 | +0.902 | 0.082 | **-0.763** | NO | OK（数値上、unseenに目視の疑義あり） |

最良seed(0)のスコア0.424は事前登録の閾値（≥1/2, <1/5）の中間。c1（A-1/Red、base入力）はseed2でv3基準に近い改善が見られたが、c2（A-2/Green、オラクル入力）は2 seedとも方向相関が負（向きが逆）。軸1（action_std）は0/3で変化なし。軸2は2/3の基準に届かずFAIL。

**総合：§21.1で期待した「両仮説とも支持される」結果には届かず。** c1改善・c2不変という非対称な結果。詳細・スクリプトの自動判定バグ（状態崩壊seedをbestとして誤選択）の訂正は`docs/v6_experiment_log.md` §22。

**① y軸符号確認・② 2×2交絡分解（2026-09-28）**：①は統計・手計算とも符号バグなし（`docs/v6_experiment_log.md` §23.1）。②はc3(r3_stay,r=A2)・c4(r2_a1random_a2rl,r=A1)を追加測定。

| seed | c1(r3_stay,A1) | c3(r3_stay,A2) | c4(r2,A1) | c2(r2,A2) |
|---|---|---|---|---|
| 0 | +0.658/73.1% | +0.002/4.2% | +0.960/84.6% | -0.527/0.4% |
| 2 | +0.902/94.7% | +0.164/48.0% | +0.929/99.6% | -0.763/2.4% |

（dirCorr / 45°以内割合）r=A1列は両データセットで高い、r=A2列は両データセットで低い〜負 → **rで決まる（Green/A-2のrに固有の困難、データセットの偏りではない）**。詳細は`docs/v6_experiment_log.md` §23.2。

**actorのr応答性検証（2026-09-28、`analyze/actor_rollout_v6.py`、§24で事前登録）**：FiLM+リラベリングなしseed0/2、各r条件50ep×100stepロールアウト。

| seed | r=A1→Red距離（初期→終端） | r=A2→Green距離（初期→終端） | actor出力差（L2/cos） |
|---|---|---|---|
| 0 | 15.07→4.51 | 13.69→**15.26（悪化）** | 2.15 / -0.21 |
| 2 | 14.65→1.04 | 14.09→7.92（改善弱い） | 4.91 / -0.52 |

**仮説（actorがrに応答しているか）は支持：actorはrに応じ出力を変えるが、r=A2でGreenへ効果的に接近できていない。** これはv6固有（単一の条件付きactorがA1/A2両方を担う設計）の問題で、A-2専用actorを使うv4の説明とは分離して記録（`docs/v6_experiment_log.md` §24.5）。

**カリキュラム学習の結果（2026-09-28、§25で事前登録、commit `fd2719c`）**：`--film --no_relabel --curriculum`、seed0/1/2。

**主指標（ロールアウト、50ep×100step、終端距離）**：

| seed | r=A1→Red | r=A2→Green | 両方≤5 |
|---|---|---|---|
| 0 | 7.39 | 6.78 | NO |
| 1 | **1.50** | **2.29** | YES |
| 2 | **1.16** | **1.73** | YES |

**主指標PASS（2/3 seed）。** 過去一貫していた「A1良好・A2不良」の非対称が解消し、seed1・2ではA1・A2がほぼ同水準に到達。

**副指標**：方向情報（c1〜c4）はc2（r2データ,A2）が3 seedとも正に転じた（以前は負）が依然最弱（dirCorr+0.06〜+0.65、seed1/2はwithin45が8%台とチャンス以下）；c3（r3_stayデータ,A2）はc1並みに改善（+0.83〜+0.88）。軸1（action_std）は不変でFAIL（0/3健全）。軸2（基準A〜D）は数値上3/3合格、目視はA1/A2良好・unseenに留保（x軸勾配の疑いが残る）。詳細は`docs/v6_experiment_log.md` §26。

**推奨**：主指標PASSを根拠にS2進行を提案。seed2が最有力候補（c1 dirCorr+0.942、ロールアウト到達度最良）。ユーザ判断待ち、S2未着手。

**c2の距離別分析（2026-09-28、§27で事前登録、`analyze/probe_q_direction_by_distance_v6.py`）**：c2データの90%が距離<5に集中（mean dist=3.33）。距離別dirCorr：

| seed | c2 <5 | c2 5-10 | c2 10-15 | c2 ≥15 |
|---|---|---|---|---|
| 0 | +0.634 | +0.700 | +0.728 | +0.915 |
| 1 | -0.023 | +0.739 | +0.858 | +0.864 |
| 2 | +0.046 | +0.737 | +0.837 | +0.856 |

5以遠では3 seedともc1/c3/c4並みの+0.70〜0.92に回復 → **範囲制限の産物、criticは健全と判定**（`docs/v6_experiment_log.md` §27.3）。S4に進んでよい。

**S2はseed2のみで実施**（v4と同じ運用：探索段階1 seed）。選定理由：ロールアウト最良（1.16/1.73）、c1 dirCorr最高（+0.942、全距離ビンで安定）。

## S2: base 段（MSE）

**実施（2026-09-28開始, 2026-09-30完了）**：commit `f223b27` + 未コミット更新分。`config/exp/v6_s2_base_mse.yml`（`model=SuperpositionNetworkProbeQV6`, probe critic=seed2 curriculum FiLM critic, r固定=A-1真値, r3_stayデータ, Encoderスクラッチ, 400ep, mse_loss, probe_criticのみfreeze, seed=1seed のみ, GPU1）。

**収束判定**（§4.2基準、`analyze/check_convergence.py` → `data/result/v6_baseline/convergence_check_v6.json`）：**CONVERGED**。trans@55, v_min(B)=16.098, m_W(fp_other)=16.501（基準(i): 16.501 ≤ 1.25×16.098=20.12 → OK）, drift=0.321（基準(ii): |slope|×|W| ≤ 0.10×16.098=1.610 → OK）。

**視覚損失（late-5, ep360-400, r3_stay/eval, in-distribution）**：self_vision = 8.7759 ± 0.1145（閾値 ≤12.05, v5_base_mse比 **PASS**）／feature_prediction_other = 16.5199 ± 0.1440（閾値 ≤20.1 **PASS**）／feature_prediction_self = 8.7011 ± 0.1896（参考値、閾値なし）。

**4軸R²（late-5、`analyze/aggregate_lateckpt.py`）**：

| データセット | h1→self | h1→other | h2→self | h2→other |
|---|---|---|---|---|
| r3_stay（in-distribution, §4.7 dual-report①） | 0.8215 ± 0.00397 | 0.5400 ± 0.00151 | 0.2061 ± 0.00586 | 0.5176 ± 0.00135 |
| r2_a1random_a2rl（canonical protocol, §4.7 dual-report②） | 0.4878 ± 0.01109 | 0.4691 ± 0.00346 | 0.0917 ± 0.00092 | 0.6713 ± 0.00249 |

h1→self（r3_stay, in-distribution）: 0.8215 ≥ v5参照値0.796 → **PASS**。

保存先：`data/result/v6_baseline/v6_s2_base_mse_s0_late5_aggregate_r2_a1random_a2rl.json`（canonicalの集計。r3_stay側の同名集計jsonは`aggregate_lateckpt.py`の出力ファイル名がデータセット名を含まない仕様のため、canonical実行時に上書きされた——数値は本行に記載済みのものが正、インシデントとして`docs/v6_experiment_log.md` §30に記録）。

**図（4点、すべて目視確認済み）**：
- 学習曲線：`data/result/v6_baseline/v6_s2_base_mse_training_curve.png`
- PCA状態マップ：`data/result/v6_baseline/pca_state_v6_s2_base_mse.png`
- 予測画像：`data/result/v6_baseline/v6_s2_base_mse_pred_images_ep400.png`
- 4軸R²棒グラフ：`data/result/v6_baseline/v6_s2_base_mse_r2_bars.png`

**総合判定：PASS**。詳細は`docs/v6_experiment_log.md` §30。

**追加検証（2026-09-30、h¹→selfのcanonical低下幅がv5より大きい件）**：詳細は`docs/v6_experiment_log.md` §31。

| 軸 | v5 in-dist | v5 canonical | v5 Δ | v6 in-dist | v6 canonical | v6 Δ |
|---|---|---|---|---|---|---|
| h¹→self | 0.7958±0.0047 | 0.7159±0.0019 | -0.080 | 0.8215±0.0040 | 0.4878±0.0111 | -0.334 |
| h¹→other | 0.5872±0.0025 | 0.4933±0.0080 | -0.094 | 0.5400±0.0015 | 0.4691±0.0035 | -0.071 |
| h²→self | 0.2203±0.0070 | 0.0827±0.0013 | -0.138 | 0.2061±0.0059 | 0.0917±0.0009 | -0.114 |
| h²→other | 0.5289±0.0008 | 0.6423±0.0029 | +0.113 | 0.5176±0.0014 | 0.6713±0.0025 | +0.154 |

probe-Q較正診断（`analyze/diag_probeq_calibration_v6.py`、v6 S2 ep400、各1000エピソード）：生Q平均 r3_stay=1.9543 / r2_a1random_a2rl=2.2187（+0.51σ相当のシフト）、飽和率P(|norm Q|>0.95) 6.97% vs 7.96%（ほぼ同等）、方向内std（D項）0.0517 vs 0.0475（8%減）。**結論：候補(b)（σ過小による飽和）は飽和率の点で支持されず、部分的（较正ミスマッチ）にとどまる。**

単一時刻プローブQ→self_position直接回帰（LSTM無し、`analyze/probe_q_direct_regression.py --dataset {r3_stay,r2_a1random_a2rl}`、v6 S2 ep400、各3000エピソード。v4参照値は`r3_a_direct_probe_q_direct_regression.json`、v3の素critic・r2_a1random_a2rl）：

| | v4 (r3_a_direct) | v6 S2 r3_stay | v6 S2 r2_a1random_a2rl |
|---|---|---|---|
| q1_vec→self_position R² | 0.5989 | 0.4051 | 0.3038 |

入力側の相対低下は-25%（0.4051→0.3038）、LSTM後（h¹→self）の相対低下は-41%（0.8215→0.4878）。入力の劣化だけでは説明できない追加の低下があり、**消費側（process-1のLSTM/デコーダ）がv3より大きな寄与をしている（増幅している）ことを示唆**。ただし事前登録した二択（「同水準→消費側確定」/「r2で明確に低い→入力側確定」）のどちらにもきれいに当てはまらず、**複合的要因として記録、確定的結論ではない**。詳細は`docs/v6_experiment_log.md` §31.5。

**独立した知見（トレードオフ）**：v6の条件付きcriticは値域の通約可能性を構造的に得る代わりに、probe-Qの自己位置特異性（single-timestep R²）を0.60（v4）→0.41（v6）まで失う。構成概念妥当性（「価値を理解したのかただ位置を読んだだけか」というv4での懸念）の観点では望ましい変化だが、位置情報の供給源としては弱くなっており、h¹→selfの低さと無関係とは言い切れない。両面を`docs/v6_experiment_log.md` §31.6に記録。

## S3: base 段（L1）

**起動（2026-09-30）**：`config/exp/v6_s3_base_l1.yml`（`v5_base_l1.yml`同一構成、pretrain=`v6_s2_base_mse` ep400）。GPU2、90秒スモークテストでpretrainロード・freezeリスト（`vision_decoder_module`のみ学習）・学習ループを確認後、本番起動。実測1.76it/s、200ep×300batch ≈ 9.5h見込み。

**結果（2026-10-01完了）**：視覚損失 self_vision L1 late-5 = 11.2111±0.0128（v5_base_l1の12.0393±0.0178より良い、PASS）。4軸R²（`superposition_module`凍結のためlate-5 sd=0.00000）：

| 軸 | v5 in-dist | v5 canonical | v6 in-dist | v6 canonical |
|---|---|---|---|---|
| h¹→self | 0.7898 | 0.7175 | 0.8220 | 0.4730 |
| h¹→other | 0.5917 | 0.4960 | 0.5388 | 0.4718 |
| h²→self | 0.2275 | 0.0810 | 0.2119 | 0.0910 |
| h²→other | 0.5292 | 0.6455 | 0.5188 | 0.6708 |

S2で確認された非対称性（h¹→selfのみcanonicalで選択的に悪化）がS3でもそのまま残存（SM凍結のため当然）。総合判定：視覚損失PASS、4軸R²は軸によりv5を上回る/下回るが既知のパターンの継続であり許容範囲。詳細`docs/v6_experiment_log.md` §34。図：`data/result/v6_baseline/v6_s3_base_l1_training_curve.png`（目視確認済み、健全）。

## S4: オラクル実験

**結果（2026-10-01）**：`analyze/oracle_eval_v6.py --exp_config v6_s3_base_l1 --epoch 200 --n_episodes 3000`（GPU1、評価のみ・学習なし）。

主判定（全体、ov_enc=other_vision）：true_r=0.2533, wrong_r=0.2519, zero=0.2525, constant=0.2490 → **true_r − zero = +0.0008（不成立、v4/v5と同種の失敗パターン）**。

§28.2の事前登録した距離別アブレーション（Greenまでの距離、self_vision L1差）：

| 距離ビン | n | true_r − zero |
|---|---|---|
| <5 | 271,283 (90.4%) | +0.0014 |
| 5-10 | 13,865 (4.6%) | -0.0016 |
| 10-15 | 9,279 (3.1%) | -0.0043 |
| ≥15 | 5,573 (1.9%) | -0.0139 |

距離とともに単調に効果が強まる（全体不成立は90.4%を占める<5ビンに支配されているため）。10-15・≥15ビンではtrue_rが4条件中単独最良。

h²のR²（true_r条件）：h²→self=0.0160、h²→other=0.8797（zeroの0.8227を上回る）。v4の「位置優先」病理（h²→self急上昇）は再現せず。

**判定（§33.3の事前登録基準を適用）：S5に進んでよい**（「遠い領域でのみ成立」の事前登録解釈規則がそのまま当てはまる）。留保：全体判定は不成立のまま、`constant`条件の主効果あり（方向情報と非零シグナルの効果を完全分離できず）、1 seedのみ。詳細`docs/v6_experiment_log.md` §35。

図：`data/result/v6_baseline/v6_s3_base_l1_oracle_ablation_bars.png`（4条件アブレーション）、`v6_s3_base_l1_oracle_r2_by_condition.png`（4軸R²）、`v6_s3_base_l1_oracle_distance_binned.png`（距離別、いずれも目視確認済み）。

## S5: VE' による推定

（未着手）

## S6: 観察の時間発展

（未着手）
