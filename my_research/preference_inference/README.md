# Preference Inference: Approach B (B-base / B-MGVE)

**「A-1 が A-2 の行動を観察することで、A-1 自身の Q 値経験を通じて A-2 の選好を推論できるか」**

Noguchi et al. (2022) の Superposition Network を拡張し、A-1 を RL エージェントとして選好を持たせた上で、共有 LSTM の内部状態 h² が A-2 の選好を符号化するかを検証する。

---

## 環境設定

```
アリーナサイズ: 20×20 (座標 -9.5 ~ +9.5)

ランドマーク:
  Red   (-9, +9)  ← A-1 が好む
  Green (-9, -9)  ← A-2 が好む（A-1 は嫌う）
  Blue  (+9, -9)
  Cyan  (+9, +9)

エージェント:
  A-1: 全方位カメラ (64×16px, RGB) + SAC で学習した RL エージェント（赤好き・緑嫌い）
  A-2: SAC で学習した RL エージェント（緑好き）← v3 から。v2 はスクリプト制御
```

### エージェントの報酬設計

| エージェント | 条件 | 報酬 |
|---|---|---|
| A-1 | 視野内の赤ピクセル数 > 50 | +1.0 |
| A-1 | 視野内の緑ピクセル数 > 50 | −1.0 |
| A-2 | 視野内の緑ピクセル数 > 50 | +1.0 |

A-1 と A-2 は**対立する選好**を持つ。

---

## ネットワーク構成

### B-base (SuperpositionNetworkApproachBBase)

```
Process-1（自己）: [sv_enc, Q¹] → SM (LSTM) → h¹
Process-2（他者）: [ov_enc,  0] → SM (LSTM) → h²
h¹・h² → Integration → so → Decoder → pred_vision
損失: MSE / L1 (pred_vision, next_vision)
```

### B-MGVE (SuperpositionNetworkApproachBMGVE)

```
Process-1（自己）: [sv_enc,  Q¹] → SM (LSTM) → h¹
Process-2（他者）: [ov_enc, Q̂²] → SM (LSTM) → h²
  ov_enc → MG (LSTM) → om_gen（2次元: A-2 の動き推定）
  ov_enc + om_gen → VE (MLP) → Q̂²（A-2 の価値推定）
  h¹・h² → FPM → pred_feature（FPM 損失で h² に視覚情報を強制）← v3 から
h¹・h² → Integration → so → Decoder → pred_vision
損失: L_vision + L_feat_self + L_feat_other
フリーズ（v3）: SM・enc・dec・Integration・FPM（MG・VE のみ更新）
```

VE への明示的な教師信号は与えない。VE は L_vision と L_feat_other の勾配が SM を通じて間接的に届くことで学習する（研究の本質: 直接教えずに推論できるかの検証）。

### 元論文 exp3（参照）

```
Process-2: [ov_enc, om] → SM (LSTM, frozen) → h²
  ov_enc → MG (LSTM, trainable) → om
  h¹・h² → FPM → pred_feature
損失: L_vision + L_feat_self + L_feat_other
フリーズ: SM・エンコーダ・デコーダ・Integration・FPM（MG のみ更新）
```

---

## 元論文との比較

| 項目 | 元論文 | v2 | v3（予定） |
|---|---|---|---|
| A-1 エージェント | ランダム移動（RL なし） | SAC（赤+1、緑−1） | SAC（赤+1、緑−1） |
| A-2 エージェント | 静止 / 周期移動 | スクリプト（ランダムランドマーク） | **SAC（緑+1）** |
| B-base 時の A-1 | ランダム移動 | RL 方策（偏った探索） | **ランダム移動** |
| FPM | あり（exp1 から） | なし | **あり** |
| フリーズ戦略 | あり | なし | **あり** |
| 訓練段階 | MSE → L1 → MG | MSE のみ | **MSE → L1 → MGVE** |
| MG への損失 | 間接（frozen SM 経由） | なし | **間接（同左）** |
| VE | なし | あり（未学習） | **あり（間接損失のみ）** |
| B-base 訓練量 | 2100ep × 101step | 100ep × 200step | **2100ep × 101step** |
| B-MGVE 訓練量 | 2100ep × 101step | 500ep × 200step | **2100ep × 101step** |
| seq_length | 101 | 200 | **101** |
| MG hidden 次元 | 128 | 64 | **128** |

---

## 訓練手順

### 元論文の手順（参照）

| ステップ | 内容 | データ | フリーズ |
|---|---|---|---|
| exp1_mse | B-base を MSE 訓練（FPM 含む） | other_stay 100ep | なし |
| exp1_l1 | L1 で再訓練（デコーダのみ更新） | 同上 | SM・enc・Integration・FPM |
| exp3 | MG 追加訓練 | other_stay_periodic **2100ep** | SM・enc・dec・Integration・FPM |

A-1 はランダム移動エージェント（RL 訓練なし）。

### v2 の手順（実施済み）

| ステップ | 内容 | データ | フリーズ |
|---|---|---|---|
| v3_rl | A-1 を SAC 訓練（1000ep×100step, 赤+1/緑−1） | — | — |
| データ収集 | B-base 用・B-MGVE 用・評価用を収集 | 下表参照 | — |
| v2_exp_b_base | B-base を MSE 訓練（FPM なし） | other_stay **100ep×200step** | なし |
| v2_exp_b_mg_ve | B-MGVE 訓練（B-base 重み引き継ぎ） | random_landmark **500ep×200step** | なし（問題点） |

#### v2 収集データ

| データセット | A-1 | A-2 | エピソード数 | 用途 |
|---|---|---|---|---|
| v2_self_rl_other_stay | RL 方策 | 静止 | 100ep × 200step | B-base 訓練 |
| v2_self_rl_other_random_landmark | RL 方策 | ランダムランドマーク | 500ep × 200step | B-MGVE 訓練 |
| v2_fixed_red/green/blue/cyan | RL 方策 | 特定ランドマーク固定 | 各 100ep × 200step | 評価 |

### v3 の手順（予定）

| ステップ | 内容 | データ | フリーズ |
|---|---|---|---|
| v3_rl_a1 | A-1 を SAC 訓練（実装済み） | — | — |
| v3_rl_a2 | A-2 を SAC 訓練（緑+1のみ） | — | — |
| データ収集 | B-base 用・B-MGVE 用・評価用を収集 | 下表参照 | — |
| v3_b_base_mse | B-base を MSE 訓練（FPM 含む） | other_stay **2100ep×101step** | なし |
| v3_b_base_l1 | L1 で再訓練（デコーダのみ更新） | 同上 | SM・enc・Integration・FPM |
| v3_b_mgve | MG・VE 追加訓練 | A-2 緑固定 **2100ep×101step** | SM・enc・dec・Integration・FPM |

#### v3 収集データ（予定）

| データセット | A-1 | A-2 | エピソード数 | 用途 |
|---|---|---|---|---|
| v3_base_train | **ランダム移動** | 静止 | 2100ep × 101step | B-base 訓練 |
| v3_mgve_train | RL 方策 | **RL 方策（緑好き）** | 2100ep × 101step | B-MGVE 訓練 |
| v3_eval | RL 方策 | RL 方策（緑好き） | 100ep × 101step | 評価 |

---

## v2 実験結果

### A-1 Q 値マップ

A-1 をグリッド状の位置に配置し、RL クリティックの Q 値を 2D マップ化。

| バージョン | Q 値範囲 | 空間コントラスト |
|---|---|---|
| v1（赤+1のみ） | 11.24〜11.45 | 差 0.2（ほぼ平坦） |
| v2（赤+1、緑−1） | 29.78〜58.08 | **差 28.3（明確な勾配）** |

v2 では赤付近で高く緑付近で低い明確な空間構造が確認された。

### 訓練損失（self_vision 再構成）

| モデル | ep1 | ep200 |
|---|---|---|
| B-base | 59.68 | 9.05 |
| B-MGVE | 23.78 | 4.97 |

損失は収束するが、評価指標（全マスク MSE）は直前フレームコピーより 10 倍以上悪く、予測品質の定量評価は不十分。

### h¹・h² → 位置の線形回帰 R²

線形回帰（Ridge）で hidden state から A-1・A-2 の位置を予測した R²。**自他分離の指標**。

| モデル | h¹ → A-1 位置 | h¹ → A-2 位置 | h² → A-1 位置 | h² → A-2 位置 |
|---|---|---|---|---|
| 元論文 exp1_l1 | **0.978** | 0.037 | 0.069 | **0.973** |
| v2 B-base | 0.770 | 0.337 | 0.508 | **0.018** |
| v2 B-MGVE | 0.849 | 0.352 | 0.878 | **0.645** |

元論文では h¹=自己専用・h²=他者専用という完全な役割分担が exp1 時点で既に実現されている。v2 では B-base 段階で h² が A-2 の位置をほぼ符号化できておらず（R²=0.018）、B-MGVE でも 0.645 にとどまる。FPM と十分なデータ量がこの分離に不可欠。

### h² の選好分類精度（ロジスティック回帰）

| モデル | 層 | 精度 |
|---|---|---|
| B-base | h¹ | 62.4% |
| B-base | h² | 31.1%（チャンス 25%） |
| B-MGVE | h¹ | 64.5% |
| **B-MGVE** | **h²** | **85.6%** |

B-MGVE の h² は A-2 の目標ランドマークを 85.6% の精度で識別できる情報を含む。ただし訓練データが異なる（A-2 スクリプト制御）ため、v3 との直接比較には注意が必要。

### MG（Motion Generator）予測精度

| A-2 目標 | MG 予測 MSE | ゼロ予測ベースライン | 倍率 |
|---|---|---|---|
| Red | 0.651 | 0.031 | 21 倍悪い |
| Green | 0.701 | 0.030 | 23 倍悪い |
| Blue | 0.778 | 0.030 | 26 倍悪い |
| Cyan | 0.657 | 0.031 | 21 倍悪い |

MG は全く学習されていない。原因: FPM がなく h² への勾配経路が存在しなかったため。

### VE（Value Estimator）出力 Q̂²

VE が正しく学習されていれば、A-2 が目標に近づくにつれ Q̂² が上昇するはず（負の相関）。

| A-2 目標 | r（Q̂² vs A-2 目標距離） |
|---|---|
| Red | +0.53（逆方向） |
| Green | +0.13（無相関） |
| Blue | +0.02（無相関） |
| Cyan | −0.09（微弱） |

Q̂² の出力スケールも訓練データの Q 値（+24〜+63）と大きくずれており（−3〜−10）、VE も未学習。

### v2 総括

| 評価項目 | 結果 |
|---|---|
| A-1 の Q 値空間構造 | ✅ 赤付近で高く緑付近で低い明確な構造 |
| h² への A-2 行動パターン符号化 | ✅ 85.6%（B-MGVE） |
| 自他分離（h²→A-2 位置 R²） | ❌ 0.018（B-base）/ 0.645（B-MGVE）← 元論文 0.973 に届かず |
| 視覚予測（self_vision）の収束 | △ 損失は収束するが定量評価不十分 |
| MG 予測精度 | ❌ ベースラインの 21 倍悪い |
| VE の Q̂² 推論 | ❌ 目標距離と無相関・スケールずれ |

**コア問題**: FPM がないため h² への勾配経路が存在せず、MG・VE が学習されていない。また B-base 段階で自他分離が実現できていないことが根本的な問題。

---

## v3 改善方針

### 設計の変更点

| 項目 | v2 | v3 |
|---|---|---|
| A-2 エージェント | スクリプト（多選好） | **SAC（緑+1のみ）** |
| 対立構造 | 不明確 | **A-1=赤好き vs A-2=緑好き** |
| B-base 時の A-1 の動き | RL 方策（偏った探索） | **ランダム移動（均一なカバレッジ）** |
| B-MGVE 時の A-1 の動き | RL 方策 | RL 方策（変更なし） |

### 必須対応

| 項目 | 内容 |
|---|---|
| **FPM 実装** | `MSE(FPM(os), ov_enc)` を追加。h² への勾配経路を確保し MG・VE を間接的に学習させる |
| **フリーズ戦略** | B-base の重みを固定し MG・VE のみ更新 |
| **データ量増加** | B-base 100ep → 2100ep、B-MGVE 500ep → 2100ep（元論文と同数） |
| **A-2 RL 化** | スクリプト制御から SAC（緑+1）へ変更 |

### パラメータ調整

| 項目 | v2 | v3 |
|---|---|---|
| seq_length | 200 | **101** |
| MG hidden 次元 | 64 | **128** |

### VE の学習方針

VE への明示的な教師信号（MSE(q2_hat, Q²)）は与えない。研究の意図は「直接教えなくても A-1 の Q 値経験から A-2 の価値が推論できるか」の検証であり、直接監督すると教師あり学習になってしまうため。

VE は以下の間接的な勾配のみで学習する:
- L_vision → pred_vision → Integration → h² → Q̂² → VE
- L_feat_other → FPM → h² → Q̂² → VE

A-2 が RL エージェントとなったことで、A-2 の行動が Q 値に基づく自然な動きとなり、VE が学習すべき価値信号が実際に存在するようになる。

### 実装順序

1. A-2 RL 訓練スクリプト（`train_rl_v3_a2.py`）実装・動作確認
2. FPM モジュール追加（model.py / modules.py）
3. フリーズ設定を config に追加
4. データ収集スクリプト更新（B-base: A-1 ランダム、B-MGVE: A-2 RL 方策）
5. MSE → L1 → MGVE の 3 段階訓練スクリプト作成
6. **B-base 完了後に h²→A-2 位置 R² を確認してから次へ進む**（分離できていなければ先に進まない）

### v3 の評価指標

| 評価 | 期待値 | 確認タイミング |
|---|---|---|
| h¹→A-1 位置 R² | **0.97 以上** | B-base 完了後 |
| h²→A-2 位置 R² | **0.97 以上**（元論文水準） | B-base 完了後 |
| 自他分離 | h¹=自己専用・h²=他者専用 | B-base 完了後 |
| VE Q̂² vs A-2 目標距離 | **負の相関**（緑に近いほど Q̂² が高い） | B-MGVE 完了後 |

---

## フォルダ構成

```
preference_inference/
├── README.md                        ← 本ファイル
├── train.py                         ← Superposition モデル訓練
├── test.py                          ← テスト・評価
├── config/
│   ├── exp/                         ← 実験設定 YAML
│   └── model/                       ← モデル設定 YAML
├── model/
│   ├── model.py                     ← B-base, B-MGVE モデル定義
│   └── modules.py                   ← SM, MG, VE, FPM モジュール
├── exp/
│   └── runner.py                    ← 訓練・評価ループ
├── simulation/
│   ├── train_rl_v2.py               ← A-1 RL 訓練（v2, 緑−2）
│   ├── train_rl_v3.py               ← A-1 RL 訓練（v3, 緑−1）
│   └── train_rl_v3_a2.py            ← A-2 RL 訓練（v3, 緑+1）← 未実装
├── analyze/
│   └── *.py                         ← 各種分析スクリプト
└── data/
    ├── model/
    │   ├── v2_rl_actor.pth
    │   ├── v3_rl_actor.pth          ← A-1 actor（実装済み）
    │   └── v3_rl_critic.pth
    ├── data/                        ← 収集済みデータセット
    └── result/                      ← 実験結果・可視化
        ├── v2_exp_b_base/
        └── v2_exp_b_mg_ve/
            └── 0/
                ├── model/           ← チェックポイント
                ├── visualize/       ← 生成した可視化図
                └── plots_fixed_v2/ ← h² PCA プロット
```

---

## 実行コマンド

```bash
# Docker に入る
docker exec -it kusano_research bash
cd /work/my_research/preference_inference

# A-1 RL 訓練（実装済み）
xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
  python -u simulation/train_rl_v3.py > /tmp/train_rl_v3.log 2>&1

# A-2 RL 訓練（v3 予定）
xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
  python -u simulation/train_rl_v3_a2.py > /tmp/train_rl_v3_a2.log 2>&1

# データ収集
bash run_collect_v2.sh

# B-base 訓練
bash run_training_v2.sh

# テスト・評価
bash run_test_v2.sh

# 分析
bash run_analysis_v2.sh
```

---

## 参照論文

Noguchi, W., Iizuka, H., Yamamoto, M., & Taguchi, S. (2022).
Superposition mechanism as a neural basis for understanding others.
*Scientific Reports*, 12, 2859.
https://doi.org/10.1038/s41598-022-06717-3
