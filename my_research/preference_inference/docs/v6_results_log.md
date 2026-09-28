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

## S2: base 段（MSE）

（未着手）

## S3: base 段（L1）

（未着手）

## S4: オラクル実験

（未着手）

## S5: VE' による推定

（未着手）

## S6: 観察の時間発展

（未着手）
