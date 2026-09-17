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
