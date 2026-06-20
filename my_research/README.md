# My Research: Preference Inference via Superposition Mechanism

## 研究概要

**「A-1 が A-2 の行動を観察することで、教師なし視覚予測のみを用いて A-2 の選好を推測できるか」**

Noguchi et al. (2022) "Superposition mechanism as a neural basis for understanding others" の実装をベースに、Shared Module (SM) の内部状態 h² が A-2 の選好を符号化するかを検証する。

元論文は自他分離・視点取得・ミラーニューロン的活性化を示した（行動パターン CW/CCW/停止）。  
本研究はその拡張として「ランドマーク選好（目的地）」という連続的・多値的な選好の表現を検証する。

---

## 環境設定

### 物理環境

```
アリーナサイズ: 20×20 (座標 -9.5 ~ +9.5)

ランドマーク（目標地点）:
  Red   (-9, +9)  ← 左上
  Green (-9, -9)  ← 左下
  Blue  (+9, -9)  ← 右下
  Cyan  (+9, +9)  ← 右上

エージェント:
  A-1: 全方位カメラ (視野 64×16px, RGB) + 全方位車輪  ← RL 訓練済み
  A-2: 全方位カメラ + 全方位車輪  ← MultiPrefAgent（選好ごとにランドマークへ向かう）
```

### A-1 の報酬設計（Step 1 で使用）

| 条件 | 報酬 |
|---|---|
| 視野内の赤ピクセル数 > 50 | +1 |
| 視野内の緑ピクセル数 > 50 | −1 |

→ A-1 は Red(-9,+9) を好み、Green(-9,-9) を嫌うエージェントに学習済み。

---

## アプローチの変遷

### Phase 0: ValueEstimator アプローチ（mg5〜mg7）→ 失敗

当初は SM の h² に **ValueEstimator (VE)** を追加して A-2 の Q 値を推定するアプローチを試みた。

```
Process-2（他者推定）:
  ov_enc + om_generated → VE → q_other（A-2 の推定 Q 値）
```

**結果と教訓:**

| バージョン | 主な変更 | 結果 | 問題点 |
|---|---|---|---|
| mg5 | VE 入力 = h²_{t-1}（循環依存） | R² 指標が逆転 | h² が A-1 の位置を予測してしまう |
| mg6 | VE 入力 = (ov_enc, om_gen)、A-2=Green のみ | Q-map に分離なし | Green のみでは汎化不要 |
| mg7 | A-2 = MultiPrefAgent（4 選好） | h² PCA 分離スコア 2.72 | VE の Q-map は分離できず、h² 自体は分離していた |

**mg7 の重要な発見**: VE による Q マップは失敗したが、SM の h² 自体が選好ごとに異なるクラスタを形成していた（分離スコア 2.72）。これが次の方針転換につながった。

### Phase 1〜3: 元論文 exp3 準拠アプローチ（現在）

VE を廃止し、**MG だけを追加**して元論文 exp3 に厳密に準拠する形に方針転換。  
h² が自然に選好情報を符号化するかを検証する。

---

## ネットワーク構成（元論文 exp3 準拠）

```
SuperpositionNetworkMotionGenerationFeaturePrediction

Process-1（自己処理）:
  v_t → encoder-1 → sv_enc  ┐
  m_t（A-1 の行動）          ├→ Shared Module (LSTM) → h¹
                              ┘

Process-2（他者推定）:
  v_t → encoder-2 → ov_enc ─→ MG (LSTM, trainable) → om_generated
                    ov_enc (masked at step>0)  ┐
                    om_generated                ├→ Shared Module (LSTM, frozen) → h²
                                               ┘

Loss:
  L_vision     = L1(pred_vision, next_vision)
  L_feat_self  = MSE(predict_feature(h¹.detach()), next_sv_enc.detach())
  L_feat_other = MSE(predict_feature(h²),           next_ov_enc.detach())
                                    ↑ detach しない → h² → SM → om → MG への勾配パス
```

**p_mask 戦略**: step>0 では sv_enc・ov_enc を 0 にマスク。SM は MG が生成した om のみを手がかりに h² を更新する。step=0 のみ本物の視覚を受け取る（初期化として機能）。

---

## Phase 1: SM の position encoding 検証 ✅

### 目的

飯塚 exp1_l1 モデルをそのまま Phase 2 の初期値として使えるかを確認する。

### 結果

| 指標 | R² | 判定 |
|---|---|---|
| h¹ → A-1 位置 | **0.9745** | ✅ 自己位置を高精度に符号化 |
| h² → A-2 位置 | **0.8060** | ✅ 他者位置もある程度符号化 |

飯塚 exp1_l1 モデルを Phase 2 の初期値として使用可能と判断。

- スクリプト: `my_research/step3/phase1/verify_iizuka_as_phase1.py`

---

## Phase 2: MG 専用訓練 ✅

### 設定（元論文 exp3 準拠）

| ハイパーパラメータ | 値 |
|---|---|
| optimizer | AdamW, lr=0.001, weight_decay=0.01 |
| batch_size | 10 |
| epochs | 200 |
| freeze | SM, encoder, decoder, feature_prediction |
| trainable | MG のみ (99,586 params) |
| Loss | L_vision + L_feat_self + L_feat_other（L_om なし） |

### 訓練データ

- A-1: RL actor, A-2: MultiPrefAgent（4 選好均等）
- エピソード数: train 2100 ep（元論文 exp3 と同数）
- 保存先: `data/data/self_random_other_stay_multi_pref/data.h5`（2492 MB）
- 収集スクリプト: `my_research/step3/collect_data_phase2.py`

### デバッグ履歴: detach バグ

v1 で MG r ≈ 0 だった原因: `os_.detach()` により `L_feat_other → h² → SM → om → MG` の勾配パスが遮断されていた。

修正: `pred_other_feat = net.predict_feature(os_)` — os_ を detach しない。

### 訓練試行の経緯

| バージョン | 設定 | MG r | 備考 |
|---|---|---|---|
| v1 (online 1000 ep) | LR=0.0001, detach バグ | ≈ 0 | MG に勾配届かず |
| v2 (online 5000 ep) | LR=0.0001, バグ修正 | 0.292 | Iizuka から開始 |
| v3 (online 5000 ep) | LR=0.001, AdamW, v2 から継続 | 0.435 | LR 変更で一時的に r 低下後に回復 |
| **offline 200 epoch** | LR=0.001, AdamW, Iizuka から開始 | **0.290** | **論文準拠の正統な方法** |

**論文準拠の方法（offline, Iizuka 開始）での結果: r=0.290**

元論文の目標 r≥0.87 に未達。A-2 の行動（ランドマーク指向移動）が元論文（単純な CW/CCW 回転）より複雑なため。  
ただし h² には選好情報が部分的に符号化されている（後述の Phase 3 参照）。

- 結果ディレクトリ: `my_research/step3/results/phase2_offline_20260616_052720/`
- best model: epoch 40（以降は val loss が横ばい）

---

## Phase 3: h² による選好推論の時間発展評価 ✅

### 目的

「A-1 が A-2 を観察し続けることで、h² が選好を徐々に符号化していくか」を時間発展で検証する。研究問いへの直接的な証拠。

- スクリプト: `my_research/step3/phase3/visualize_phase3.py`
- 評価: 4 選好 × 80 ep × 80 steps = 25,600 サンプル
- 各 step t の h² から Logistic Regression で選好を分類（精度の時間推移）

### 主要結果

**h² の選好分類精度（時間推移）**

| 観察ステップ t | 分類精度 |
|---|---|
| t=0（観察開始直後） | 0.271（≈ chance 0.25） |
| t=20 | 0.625 |
| t=40 | 0.708 |
| t=65（最大） | **0.771** |
| t=79（最終） | 0.760 |

→ **観察開始時は chance レベルだが、観察継続で最大 0.77 に到達。**

**混同行列（全ステップ, acc=0.659）**

| 選好 | 分類精度 | 解釈 |
|---|---|---|
| **Cyan (+9,+9)** | **87.4%** | A-1 の定位置（Red 付近）から最も遠く、A-2 の移動方向が最も明確 |
| Red (-9,+9) | 60.3% | A-1 自身が Red 付近に留まるため、A-2 の動きが視野内で判別しにくい |
| Green (-9,-9) | 57.9% | Red と左側・y 方向のみ異なり混同されやすい |
| Blue (+9,-9) | 56.3% | Red と対角、A-1 との相対位置変化が類似する場合がある |

**PCA 構造**: PC1 寄与率 85.3%（h² の変動のほぼ全体を 1 次元で説明）

### 解釈

1. **視点依存の選好推論**: Cyan が最も精度良く推論されるのは、A-1 が Red 付近を好み（A-1 の視点から）Cyan 方向への A-2 の移動が最も識別しやすいため。「A-1 の視点から見た他者理解」という元論文の主張と整合する。

2. **時間発展の急激な向上**: step=0〜20 で急激に精度が上昇する。A-2 の移動方向が確定し始めるタイミングと一致する。

3. **MG r の低さと h² 符号化の乖離**: MG r=0.290 は目標に届かないが、h² clf acc=0.77 は十分な選好符号化を示す。SM が不完全な MG 推定からでも選好情報を h² に抽出できていることを示唆する。

---

## 現在の結果まとめ

| Phase | 指標 | 値 |
|---|---|---|
| Phase 1 | h² → A-2 位置 R² | 0.806 |
| Phase 2 | MG Pearson r | 0.290（目標 ≥0.87 未達） |
| Phase 3 | h² 選好分類精度（t=65） | **0.771** |
| Phase 3 | h² 選好分類精度（t=0） | 0.271（≈ chance） |

**コアメッセージ**: 観察開始時（t=0）は chance レベルだった選好推論精度が、観察継続（t=65）で 0.77 に達する。A-1 の SM 内部状態 h² は A-2 の観察を通じて選好情報を漸進的に符号化する。

---

## 次のステップ

### 1. Phase 3 の可視化 PNG の確認・改善

現在の `phase3.png` に追加できる可視化:
- t-SNE による非線形クラスタ可視化
- 選好別の PC1 アトラクター安定性（収束先の分散）

### 2. Phase 1 の自前訓練（Option B: 独自改良、論文方針からの逸脱）

現在の配布ギャップ（Iizuka: A-2 静止 → Phase 2: A-2 移動）が MG r 低さの原因の一つ。  
ただし元論文は Phase 1 で A-2 を静止させるため、**変更は論文方針からの逸脱**になる。  
「独自の工夫として」試す価値はある。

### 3. 論文執筆へ向けた整理

- Phase 3 の結果（t=0→0.77 の時間発展）を Figure として整理
- 元論文との比較（行動タイプ推論 vs 選好推論）
- MG r が低い理由を考察（タスクの複雑さの違い）として明示

---

## フォルダ構成

```
my_research/
│
├── README.md                       ← 本ファイル
├── rl_model_v6_actor.pth           ← A-1 訓練済み Actor (SAC+LSTM)
├── rl_model_v6_critic.pth          ← A-1 訓練済み Critic (SAC+LSTM)
├── rl_agent_sac.py                 ← ActorLSTM / CriticLSTM 定義
├── agent2_multi_pref.py            ← MultiPrefAgent / LandmarkFollowerAgent
├── agent2_green.py                 ← GreenFollowerAgent（旧 mg6 まで使用）
├── train_rl_v6.py                  ← Step 1 学習スクリプト（最終版）
│
├── step2/                          ← VE/Q値アプローチ（mg5 まで）
│   └── ...
│
└── step3/
    ├── collect_data_phase2.py          ← offline データ収集
    ├── value_estimator_v2.py           ← VE V2（旧 mg6/mg7 用）
    │
    ├── mg6/                            ← VE 改良版（A-2=Green）
    │   └── ...
    ├── mg7/                            ← 多選好版（h² PCA 分離スコア 2.72 を確認）
    │   └── ...
    │
    ├── phase1/
    │   └── verify_iizuka_as_phase1.py
    │
    ├── phase2/
    │   ├── train_step3_phase2.py           ← online 訓練（試行用）
    │   ├── train_step3_phase2_offline.py   ← offline 訓練（論文準拠・主要）
    │   └── visualize_phase2.py             ← Phase 2 評価
    │
    ├── phase3/
    │   └── visualize_phase3.py             ← Phase 3 評価（時間発展）
    │
    └── results/
        ├── mg6_20260606_191630/            ← mg6 結果（VE アプローチ）
        ├── mg7_20260608_154447/            ← mg7 結果（h² PCA 分離 2.72）
        ├── phase1_20260615_054928/         ← Phase 1 検証
        ├── phase2_20260615_182451/         ← v1: detach バグ (r≈0)
        ├── phase2_20260615_193019/         ← v2: detach 修正 (r=0.292)
        ├── phase2_20260616_004705/         ← v3: LR=0.001 (r=0.435)
        └── phase2_offline_20260616_052720/ ← offline 200 epoch【主要結果】
            ├── model_best.pth              ← best model (epoch 40)
            ├── train.log
            ├── visualize_phase2.png        ← Phase 2 評価図
            └── phase3.png                  ← Phase 3 評価図
```

---

## 実行コマンド

```bash
docker exec -it kusano_research bash
cd /work
Xvfb :99 -screen 0 1024x768x24 & sleep 1
```

### Phase 3 評価（メイン）

```bash
DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \
  python3 my_research/step3/phase3/visualize_phase3.py \
    --model my_research/step3/results/phase2_offline_20260616_052720/model_best.pth \
  2>&1 | tee my_research/step3/results/phase2_offline_20260616_052720/phase3.log
```

### Phase 2 offline 訓練（論文準拠）

```bash
DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \
  python3 my_research/step3/phase2/train_step3_phase2_offline.py \
  2>&1 | tee my_research/step3/results/train_phase2_offline.log
```

### Phase 2 データ収集

```bash
DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \
  python3 my_research/step3/collect_data_phase2.py \
  2>&1 | tee my_research/step3/results/collect_phase2.log
```

---

## mg7 の知見（VE アプローチ最終版）

h² PCA の選好分離スコア 2.72（クラスタ間距離 / within-std）が得られた。これが元論文 exp3 準拠アプローチへの方針転換の契機。

| 指標 | mg7 |
|---|---|
| h² 分離スコア | 2.72 |
| Green - Cyan 間距離 | 1.217（最大） |
| Red - Blue 間距離 | 0.083（最小：混同しやすい対角ペア） |

VE（Q マップ）では選好の弁別に失敗したが、h² 自体が選好情報を符号化している可能性が示された。

---

## 参照論文

Noguchi, W., Iizuka, H., Yamamoto, M., & Taguchi, S. (2022).  
Superposition mechanism as a neural basis for understanding others.  
*Scientific Reports*, 12, 2859.  
https://doi.org/10.1038/s41598-022-06717-3
