# v7 数値の一次記録

各数値に、実験ID・git commit・シード・データ（split）・使った真値を付ける。解釈は `docs/v7_experiment_log.md` に書く。

---

## 段階0-1：関門ロールアウト（2026-10-08）

- 実験ID：`v7_s0_gate_rollout_seed1`
- スクリプト：`analyze/actor_rollout_v7.py`（`actor_rollout_v6.py` のコピー、4目標に拡張）。commit：`1e5f6fe` 上の未コミット作業ツリー（段階0の完了時にコミットする）
- actor：`data/model/v6_rl_actor_seed1_film_norelabel_curriculum.pth`（FiLM）
- 環境：`self_random_other_stay`、actor を A-1 側カメラで動かし、相手は静止（v6 §25.2 と同じ方法）。50ep × 100step、`env.reset()` は位置の上書きなし
- シード：SEED=0（torch・numpy、run1/run2 は Python `random` も）
- 使った真値：なし（環境内のロールアウトのみ）。データセットは使わない
- 出力：`data/result/v7_irl/stage0/actor_rollout_seed1_film_norelabel_curriculum_{run0_unseeded_random,run1,run2}.json`、ログ `logs/v7_stage0_actor_rollout_seed1{,_run1,_run2}.log`

**ランドマークの確認**（各角の 1.5 内側に A-1 を置いて描画、色の割合 red/green/blue/cyan）：4か所とも、その角の色が最大（0.0879、他は 0.0127 以下）。座標 赤(−9,9)・緑(−9,−9)・青(9,−9)・シアン(9,9) は正しい。

**目標までの距離（主表：run1。run2 は run1 と bit-identical、md5 一致）**

| 目標 | r | 初期距離 | 終端距離 | 最小距離 | 終端で最も近いランドマーク | 判定（終端 ≤ 5） |
|---|---|---|---|---|---|---|
| 赤 | (+1,−1,0,0) | 14.07 | **2.57 ± 4.63** | 1.97 ± 3.88 | 赤 2.57 | 合格 |
| 緑 | (−1,+1,0,0) | 14.14 | **2.87 ± 4.65** | 2.45 ± 4.20 | 緑 2.87 | 合格 |
| 青 | (0,0,+1,−1) | 13.72 | **17.21 ± 2.44** | 11.75 ± 3.69 | **緑 2.82** | **不合格** |
| シアン | (0,0,−1,+1) | 14.24 | **2.32 ± 3.24** | 1.87 ± 2.81 | シアン 2.32 | 合格 |

**run0（Python `random` 未固定、v6 スクリプトと同じシード設定）**：赤 2.31 ± 3.53、緑 2.67 ± 3.68、青 17.03 ± 2.41（緑まで 2.28）、シアン 1.95 ± 2.69。判定は run1 と同じ。

**v6 §26.1 との照合**：v6 の seed1 は 赤 1.50、緑 2.29（初期距離 12.98 / 13.83）。初期距離から既に違う。初期位置は Python `random`（`simulation/simulation/agent/field_object.py`）で決まり、v6 のスクリプトはこれを固定していなかったため、v6 の値はもともと bit-identical には再現できない。3回とも、赤・緑の終端距離は 1.5〜2.9 の範囲で関門を大きく下回っている。

---

## 段階0-2：少量での bit-identical 確認（2026-10-08）

- 実験ID：`v7_s0_repro_check`
- スクリプト：`simulation/collect_data_v7.py --dataset v7_repro_check`（train 6 / test 3、各 101 フレーム、目標は各2/各1）、GPU0、2回実行
- シード：seed_key `v7_repro_check` → md5 由来のシード（random・numpy・torch）、目標の割り当ては `md5(v7_repro_check/goals/<mode>)`
- 結果：**16個のデータセット（train/test × 8 フィールド）すべて一致、属性も一致、HDF5 ファイル全体の md5 も一致**（`0359dd881a62846ae0ec3599b38a8d60`）
- 確認後にデータは削除（`data/result/v7_irl/stage0/repro/`）。ログ：`logs/v7_stage0_repro_run{1,2}.log`

## 段階0-3：本収集とデータ統計（2026-10-08）

- 実験ID：`v7_s0_collect`
- データ：`data/data/v7_a1random_a2goal3/data.h5`（2.06GB、train 3000 / test 300、各 101 フレーム）。seed_key `v7_a1random_a2goal3` → seed 25712908
- A-2 の actor：`data/model/v6_rl_actor_seed1_film_norelabel_curriculum.pth`（確率的 `sample()`、hidden を持ち越し、位置はアリーナ境界 ±10 でクリップ）。A-1：RandomAgent
- 所要：19:53〜21:07（約74分、GPU0）。ログ：`logs/v7_stage0_collect_v7_a1random_a2goal3.log`
- 統計：`analyze/dataset_stats_v7.py`。出力 `data/result/v7_irl/stage0/dataset_stats_v7_a1random_a2goal3.{json,png}`、ログ `logs/v7_stage0_dataset_stats.log`
- 使った真値：A-2 の位置・行動・目標ラベル（評価・統計のみ）

**目標ごとの統計**（距離は A-2 から目標ランドマークまで。終端・最小は平均 ± sd）

| split | 目標 | エピソード | 初期距離 | 初期距離ビン <5 / 5-10 / 10-15 / ≥15 | 終端距離 | 最小距離 | 終端 ≤5 のエピソード | 5以内のフレーム | \|a\|<0.1 のフレーム |
|---|---|---|---|---|---|---|---|---|---|
| train | 赤 | 1000 | 13.83 ± 5.71 | 80 / 192 / 258 / 470 | 1.41 ± 0.00 | 0.94 ± 0.21 | 1.000 | 0.899 | 0.0000 |
| train | 緑 | 1000 | 14.02 ± 5.48 | 69 / 172 / 281 / 478 | 1.41 ± 0.03 | 0.97 ± 0.19 | 1.000 | 0.912 | 0.0000 |
| train | シアン | 1000 | 14.04 ± 5.68 | 81 / 174 / 254 / 491 | 1.41 ± 0.01 | 0.99 ± 0.16 | 1.000 | 0.913 | 0.0000 |
| test | 赤 | 100 | 14.37 ± 4.87 | 4 / 18 / 25 / 53 | 1.41 ± 0.00 | 0.92 ± 0.25 | 1.000 | 0.892 | 0.0000 |
| test | 緑 | 100 | 12.65 ± 5.38 | 7 / 23 / 31 / 39 | 1.41 ± 0.02 | 0.98 ± 0.17 | 1.000 | 0.922 | 0.0000 |
| test | シアン | 100 | 14.77 ± 5.68 | 5 / 18 / 25 / 52 | 1.41 ± 0.00 | 1.01 ± 0.11 | 1.000 | 0.909 | 0.0000 |

- §2.6 の停止条件（収集データでシアンの平均終端距離 ≤ 5）：train・test とも満たす（停止しない）。
- 終端距離 1.41 は、A-2 がアリーナの角（±10, ±10）に押し付けられた位置とランドマーク（±9, ±9）の距離 √2。
- 図（目視確認済み）：初期距離の分布は3目標でほぼ同じ形。どの目標も 20〜25 フレームまでにほぼ全エピソードが 5 以内に入り、以後は角に張り付く。
