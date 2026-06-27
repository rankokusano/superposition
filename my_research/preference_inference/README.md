# Preference Inference: Approach B (B-base / B-MGVE)

**「A-1 が A-2 の行動を観察することで、A-1 自身の Q 値経験を通じて A-2 の選好を推論できるか」**

Noguchi et al. (2022) の Superposition Network を拡張し、A-1 を RL エージェントとして選好を持たせた上で、共有 LSTM の内部状態 h² が A-2 の選好を符号化するかを検証する。

---

## 環境設定

```
アリーナサイズ: 20×20 (座標 -9.5 ~ +9.5)

ランドマーク:
  Red   (-9, +9)  ← A-1 が好む
  Green (-9, -9)  ← A-1 が嫌う
  Blue  (+9, -9)
  Cyan  (+9, +9)

エージェント:
  A-1: 全方位カメラ (64×16px, RGB) + SAC で学習した RL エージェント
  A-2: 特定ランドマークへ向かうスクリプト制御エージェント
```

### A-1 の報酬設計

| 条件 | 報酬 |
|---|---|
| 視野内の赤ピクセル数 > 50 | +1.0 |
| 視野内の緑ピクセル数 > 50 | −1.0 |

---

## ネットワーク構成

### B-base (SuperpositionNetworkApproachBBase)

```
Process-1（自己）: [sv_enc, Q¹] → SM (LSTM) → h¹
Process-2（他者）: [ov_enc,  0] → SM (LSTM) → h²
h¹・h² → Integration → so → Decoder → pred_vision
損失: L1(pred_vision, next_vision)
```

### B-MGVE (SuperpositionNetworkApproachBMGVE)

```
Process-1（自己）: [sv_enc,  Q¹  ] → SM (LSTM) → h¹
Process-2（他者）: [ov_enc, Q̂²  ] → SM (LSTM) → h²
  ov_enc → MG (LSTM)  → om_gen (2次元: A-2 の動き推定)
  ov_enc + om_gen → VE (MLP) → Q̂² (A-2 の価値推定)
h¹・h² → Integration → so → Decoder → pred_vision
損失: L1(pred_vision, next_vision)  ← MG・VE への明示的損失なし（v2 の問題点）
```

### 元論文 exp3 (参照)

```
Process-2: [ov_enc, om] → SM (LSTM, frozen) → h²
  ov_enc → MG (LSTM, trainable) → om
  h¹・h² → FPM → pred_feature（FPM 損失で h² に視覚情報を強制）
損失: L_vision + L_feat_self + L_feat_other
フリーズ: SM・エンコーダ・デコーダ・Integration・FPM（MG のみ更新）
```

---

## 訓練手順

### 元論文の手順（参照）

| ステップ | 内容 | データ | フリーズ |
|---|---|---|---|
| exp1_mse | B-base 相当をゼロから MSE 訓練（FPM 含む） | other_stay 100ep | なし |
| exp1_l1 | L1 で再訓練（デコーダのみ更新） | 同上 | SM・エンコーダ・Integration・FPM |
| exp3 | MG 追加訓練 | other_stay_periodic **2100ep** | SM・エンコーダ・デコーダ・Integration・FPM |

A-1 は**ランダム移動エージェント**（RL 訓練なし）。

### 今回の手順 (v2)

| ステップ | 内容 | データ | フリーズ |
|---|---|---|---|
| v3_rl | A-1 を SAC で訓練（1000ep×100step） | — | — |
| データ収集 | B-base 用・B-MGVE 用・評価用を別々に収集 | 下表参照 | — |
| v2_exp_b_base | B-base を MSE 訓練（FPM なし） | other_stay **100ep** | なし |
| v2_exp_b_mg_ve | B-MGVE 訓練（B-base の重みを引き継ぎ） | random_landmark **500ep** | **なし（問題点）** |

#### 収集データ一覧

| データセット | A-2 の行動 | エピソード数 | 用途 |
|---|---|---|---|
| v2_self_rl_other_stay | 静止 | 100ep × 200step | B-base 訓練 |
| v2_self_rl_other_random_landmark | ランダムにランドマークへ | 500ep × 200step | B-MGVE 訓練 |
| v2_fixed_red/green/blue/cyan | 特定ランドマークへ固定 | 各 100ep × 200step | 評価 |

---

## v2 実験結果

### A-1 Q値マップ

A-1 をグリッド状の位置に配置し、RL クリティックの Q 値を 2D マップ化。

| バージョン | Q 値範囲 | 空間コントラスト |
|---|---|---|
| v1（赤+1のみ） | 11.24〜11.45 | 差 0.2（ほぼ平坦） |
| v2（赤+1、緑−1） | 29.78〜58.08 | **差 28.3（明確な勾配）** |

v2 では赤付近で高く、緑付近で低い明確な空間構造が確認された。

### 訓練損失（self_vision 再構成）

| モデル | ep1 | ep200 |
|---|---|---|
| B-base | 59.68 | 9.05 |
| B-MGVE | 23.78 | 4.97 |

収束は確認できるが、評価指標（全マスク MSE）は直前フレームコピーより 10 倍以上悪く、予測品質の定量評価は不十分。

### h² の選好分類精度（ロジスティック回帰）

| モデル | 層 | 精度 |
|---|---|---|
| B-base | h¹ | 62.4% |
| B-base | h² | 31.1%（チャンス 25%） |
| B-MGVE | h¹ | 64.5% |
| **B-MGVE** | **h²** | **85.6%** |

B-MGVE の h² は A-2 の目標ランドマークを 85.6% の精度で識別できる情報を含む。これは A-1 が A-2 の視覚に直接アクセスせず、共有 LSTM を通じて A-2 の行動パターンを h² に符号化できていることを示す。

### MG（Motion Generator）予測精度

| A-2 目標 | MG 予測 MSE | ゼロ予測ベースライン | 倍率 |
|---|---|---|---|
| Red | 0.651 | 0.031 | 21 倍悪い |
| Green | 0.701 | 0.030 | 23 倍悪い |
| Blue | 0.778 | 0.030 | 26 倍悪い |
| Cyan | 0.657 | 0.031 | 21 倍悪い |

MG は全く学習されていない。原因: 明示的な損失関数を設定しなかったため。

### VE（Value Estimator）出力 Q̂²

VE が正しく学習されていれば、A-2 が目標に近づくにつれ Q̂² が上昇するはず（負の相関）。

| A-2 目標 | r（Q̂² vs A-2 目標距離） |
|---|---|
| Red | +0.53（逆方向） |
| Green | +0.13（無相関） |
| Blue | +0.02（無相関） |
| Cyan | −0.09（微弱） |

Q̂² の出力スケールも訓練データの Q 値（+24〜+63）と大きくずれており（−3〜−10）、VE も未学習。

### 総括

| 評価項目 | 結果 |
|---|---|
| A-1 の Q 値空間構造 | ✅ 赤付近で高く緑付近で低い明確な構造 |
| h² への A-2 行動パターン符号化 | ✅ 85.6%（B-MGVE） |
| 視覚予測（self_vision）の収束 | △ 損失は収束するが定量評価不十分 |
| MG 予測精度 | ❌ ベースラインの 21 倍悪い |
| VE の Q̂² 推論 | ❌ 目標距離と無相関・スケールずれ |

**コア問題**: MG・VE への明示的な損失関数がなく、FPM も未実装のため、Approach B の核心部分（Q 値経験を通じた選好推論）が検証できていない。

---

## v3 改善方針

### 必須対応

| 項目 | 内容 |
|---|---|
| **FPM 実装** | `MSE(FPM(os), ov_enc)` を追加。os への直接勾配を確保し MG を間接的に学習させる |
| **フリーズ戦略** | B-base の重みを固定し MG+FPM のみ更新（元論文 exp3 に準拠） |
| **データ量増加** | B-base 100ep → 2100ep、B-MGVE 500ep → 2100ep（元論文と同数） |
| **データ収集時に Q¹ を保存** | A-2 の位置での Q¹ 値を記録（VE 損失の正解値として使用） |

### 訓練手順の改善（元論文に準拠）

| ステップ | 内容 | フリーズ |
|---|---|---|
| v3_b_base_mse | B-base を MSE で訓練（FPM 含む） | なし |
| v3_b_base_l1 | L1 で再訓練 | SM・エンコーダ・Integration・FPM |
| v3_b_mgve | MG+FPM を追加訓練 | SM・エンコーダ・デコーダ・Integration・FPM |

### その他パラメータ調整

| 項目 | v2 | v3 |
|---|---|---|
| seq_length | 200 | 101（元論文に合わせる） |
| MG hidden 次元 | 64 | 128（元論文に合わせる） |
| VE 評価方法 | Q̂² vs A-2 距離の相関 | Q̂² の空間マップ（位置ごとの平均） |

### VE への損失追加（FPM 後に VE が学習されなかった場合）

```
MSE(q2_hat, Q¹(A-2 の位置))
```

A-2 がスクリプト制御で Q 値を持たないため、A-1 の Q 関数で代用する。

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
│   └── train_rl_v3.py               ← A-1 RL 訓練（v3, 緑−1）
├── analyze/
│   └── *.py                         ← 各種分析スクリプト
└── data/
    ├── model/
    │   ├── v2_rl_actor.pth
    │   ├── v3_rl_actor.pth          ← 最新 A-1 actor
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

# A-1 RL 訓練
xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
  python -u simulation/train_rl_v3.py > /tmp/train_rl_v3.log 2>&1

# データ収集
bash run_collect_v2.sh

# B-base 訓練
bash run_training_v2.sh

# B-MGVE 訓練
# run_training_v2.sh の設定を v2_exp_b_mg_ve に変更して実行

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
