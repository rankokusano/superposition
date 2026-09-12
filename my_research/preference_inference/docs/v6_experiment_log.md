# v6 実験ログ（唯一の参照元）

**このファイルが v6 の唯一の継続的記録です。** チャットは auto compact で失われるため、設計判断・棄却された仮説・数値・未解決の懸念は必ずここに書く。新しい Claude Code セッションはこのファイルと `docs/v6_instructions.md` を読めば続きから作業できることを目標にする。

v4/v5 の記録（`docs/v4_experiment_log.md` 等）は読み取り専用。ここには書き込まない。

---

## 0. 背景（`docs/v6_instructions.md` §0 要約）

v4/v5 で判明したこと（確定・変更しない）：
- (a) A-1 の視覚予測：できる（h¹→self 0.907）
- (b) A-2 の視覚予測（base 段）：できる（h²→other 0.95〜0.98）
- (c) **A-2 の価値予測：できない**。原因は2つ、いずれも構造的
  1. Q値が自他で通約不可能（A-1 std 0.63 vs A-2 std 0.009。値域を揃えても益にならなかった＝P1.75）
  2. 位置を流すほうが視覚予測の損失に有利（`Q̂²→A-1位置 0.387 ≫ →A-2位置 0.097`、v5_r4_move）

v6 の狙い：推定対象を Q値（8次元）から**報酬パラメータ（4次元）**に変える。教授のメモの「ステップ2（逆強化学習）」に相当。狙いは上の2問題を設計レベルで同時に解消すること：
- 報酬は同じ尺度で定義される → 通約可能性が構造的に保証される
- 4次元しか出力できない → 位置情報を符号化する帯域がない
- 報酬を間違えると Q値が全部狂う → VE に「正しく推定する動機」が生まれる（v4/v5にはなかった）

失敗しても「報酬レベルまで降りても他者の価値は推定できない」という v4/v5 より強い境界の主張になる。**どちらに転んでも論文に書ける。**

## 1. 引き継ぐ研究設定（変更禁止、`docs/v6_instructions.md` §1 参照）

- アリーナ四隅：Green(-9,-9) / Red(-9,9) / Blue(9,-9) / Cyan(9,9)（コード上 "Yellow"、実測RGB(0,240,240)）
- A-1 選好=Red、A-2 選好=Green。単一固定、変更しない（過去に多選好を試みて教授の指示で撤回した経緯あり。再提案しない）
- 現行報酬：`A-1: red_fraction - green_fraction` / `A-2: green_fraction - red_fraction`
- A-1 はランダム歩行（RandomAgent）。Q値は critic から事後計算。教授の明示的指示

## 2. v6 の設計要約（`docs/v6_instructions.md` §2 参照）

```
v4/v5:  VE → Q̂²（8次元）を直接出力 → SM
v6:     VE' → r̂²（報酬パラメータ 4次元）→ 報酬条件付き critic Q(s,a,r̂²) → 8方向プローブ → Q̂²（8次元）→ SM
```

- 報酬パラメータ `r = (w_red, w_green, w_blue, w_cyan)`。A-1真値`(+1,-1,0,0)`、A-2真値`(-1,+1,0,0)`
- **SM のアーキテクチャは変更しない**（v4/v5との比較可能性を保つ）
- 新規実装の中心：**報酬条件付き critic** `Q(s,a,r)`。SAC学習中にエピソードごとに`r`をランダムサンプルして学習
- Encoder は v5 と同じくスクラッチ学習（元論文Encoder凍結では収束しないことがv5で判明済み）
- 主たる比較対象：`v5_r4_move`（Encoderの条件を揃え、違いを「推定対象」と「critic形式」だけにする）
- 段階：S1（条件付きcritic学習）→ S2（base MSE）→ S3（base L1）→ S4（オラクル実験・最重要ゲート）→ S5（VE'推定）→ S6（時間発展）
- 各段の判定基準は `docs/v6_instructions.md` §3 に事前登録済み。結果を見てから基準を作らない

---

## 3. 設計判断ログ

判断のたびに1段落で追記する。フォーマット：**日付 / 判断 / 理由 / 代替案とその却下理由**。

### 2026-09-12: `my_research/rl_agent_sac.py` は直接編集せず、`preference_inference/` にコピーしてから拡張する

**判断**：v6 の報酬条件付き `CriticLSTM` / `ActorLSTM` は、共有ファイル `my_research/rl_agent_sac.py` を直接書き換えるのではなく、`my_research/preference_inference/` 配下にコピーした上で拡張する。

**理由**：`rl_agent_sac.py` は `my_research/step3/`（別研究ライン、触ってはいけない、§5.3）を含む70以上のファイルから import されている共有モジュールであることを調査フェーズで確認した（詳細は §4 調査結果を参照）。ここを直接変更すると step3 側の再現性を壊すリスクがある。作業ルール「Work ONLY in my_research/preference_inference/」（既存メモリ）とも整合する。

**代替案**：直接編集 → step3 の凍結資産（`model_best.pth` 等）を壊すリスクがあるため却下。

---

## 4. 調査フェーズの結果（§7、コード変更なし）——2026-09-12

### 4.1 現行報酬関数は4次元形式の特殊ケースか

**確認：YES。** `my_research/preference_inference/simulation/train_rl_v3.py` の `get_reward()`:

```python
def get_reward(vision_np):
    r, g, b = vision_np[:, :, 0], vision_np[:, :, 1], vision_np[:, :, 2]
    red_pixels   = int(((r > 0.9) & (g < 0.1) & (b < 0.1)).sum())
    green_pixels = int(((g > 0.9) & (r < 0.1) & (b < 0.1)).sum())
    ...
    return red_fraction - green_fraction
```

これは `reward = Σ_k w_k · fraction_k`（`w=(+1,-1,0,0)`, k∈{red,green,blue,cyan}）の特殊ケース。A-2用（`train_rl_v3_a2.py`）も対称で `w=(-1,+1,0,0)` に一致。

**拡張に必要なもの**：blue・cyan の pixel fraction 計算が現状コードベースに存在しない（`blue_pixels`/`cyan_pixels` 相当は grep で0件）。しきい値は red/green と同じパターン（該当色チャンネル>0.9、他<0.1）で機械的に追加できるはずだが、**cyan ランドマークは "Yellow" マテリアル（Kd=(0.8,0.8,0)、field.mtl）なのに実測RGBがシアンだった**という v4 の既知の食い違いがある（`v4_experiment_log.md` §2.1 該当箇所）。実装前に实際の `other_vision`/`self_vision` データでピクセルのRGBヒストグラムを取り、4色のしきい値を実測で確定させる必要がある（1時間程度の作業、大きなコストではない）。

**結論**：構造的には自明な拡張。**実装コストは小**（既存パターンの複製＋しきい値の実測確認）。

### 4.2 SAC critic を報酬条件付きに拡張する実装コスト

現行実装：`my_research/rl_agent_sac.py`（88行）の `CriticLSTM`：
```python
self.q = nn.Sequential(nn.Linear(hidden_dim + 2, 64), nn.ReLU(), nn.Linear(64, 1))
def forward(self, v, action, hidden=None):
    ...
    x = torch.cat([lstm_out, action], dim=-1)
    q = self.q(x)
```

**ネットワーク自体の変更は小さい**：`nn.Linear(hidden_dim + 2, 64)` → `nn.Linear(hidden_dim + 2 + 4, 64)`、`forward` に `r` 引数を追加して `torch.cat([lstm_out, action, r], dim=-1)` にするだけ。

**訓練ループ（`train_rl_v3.py`、227行）側の変更点**（中程度のコスト）：
1. エピソード開始時に `r` をランダムサンプル（サンプル分布は未決定 — §5 未解決の懸念を参照。**これは設計判断であり、結果を見る前に固定すべき事項**）
2. 報酬計算を4次元一般形に変更（4.1のfraction計算を利用）
3. `ReplayBuffer` の transition に `r`（4次元）を追加保存
4. ミニバッチ更新時、`critic1/2`・`critic1/2_target` の全呼び出しに `r` を渡す
5. **Actor も `r` を条件に含めるべきか要検討**：現行 `ActorLSTM` は `r` を見ない。SACのターゲットQ計算 `actor.sample(next_states)` は特定の `r` の下での方策を評価するので、Actorが`r`を無視すると全`r`の平均的方策に収束し、critic学習の質が下がる可能性がある。**これは未決定の設計判断**（§5参照）。

**周辺コードへの波及**（軽微〜中程度）：
- `model/model.py` の `compute_probe_q()`（2箇所、301行台・459行台）：`self.probe_critic(v_rep, a_rep, hidden=None)` → `r_rep` を追加で渡す必要。呼び出し元（VE'または オラクル実験でのr固定値）から`r`が来る経路を新設する必要がある
- `analyze/check_critic_health.py`：健全性チェックに `r` を通す引数を追加し、`r=(+1,-1,0,0)` と `r=(-1,+1,0,0)` の両方でチェックする必要（§3.0のS1判定基準が要求）

**重要な発見（実装コストに直結）**：`my_research/rl_agent_sac.py` は `my_research/preference_inference/` 以外に **`my_research/step3/`（別研究ライン、保護対象）を含む70以上のファイルから import されている共有モジュール**。直接編集すると step3 に影響しうるため、**`my_research/preference_inference/` 配下にコピーしてから拡張する**方針とした（§3の設計判断ログ参照）。コピー自体はコストではなくむしろ安全策。

**総合見積もり**：ネットワーク変更自体は数十分〜1時間。訓練ループ・health check・model.py配線を含めた実装全体で**半日〜1日**（設計判断が先に固まっていることが前提）。指示書 §6.1 の見積もり（S1に5〜10h、うち大半は学習時間そのもの）と整合的。

**微分可能な価値計算が必要か、条件付きcriticで足りるか**：指示書の設計（VE'→r̂²→条件付きcritic→8方向プローブ→Q̂²→SM、§2.1）は、**critic自体は凍結して使う**（S2以降、生成段でcriticは学習しない、§3.0の表「生成モジュール：VE'のみ学習」）。したがって学習時にはVE'の勾配がcriticの中を通ってprobe Q̂²まで伝播する必要があり、**criticはPyTorchモジュールとして微分可能である必要がある**（数値ブラックボックスでは不可）。現行`CriticLSTM`はニューラルネットなのでこの要件は自動的に満たされる。追加の微分可能性対応は不要。

### 4.3 §10.2 の図の材料

**(a) other_vision loader パッチ**：現在のブランチ分岐元（`feature/v4-value-superposition`、未コミットの作業ツリー変更を含む）で**既に適用済み**であることを確認した。`exp/loader.py`・`exp/runner.py` の未コミット diff に `other_vision` 関連のパッチが含まれている（`SequenceLoader`/`SeqLoader` の `__getitem__` で `other_vision` を読み込み、`runner.py` の modal リストにも追加済み）。**v6ブランチはこの状態を引き継いでいるので追加作業は不要**。

**(b) viz20 データセット**：Docker コンテナ内で実データを確認。
```
data/data/r2_a1random_a2rl_viz20/data.h5  train/{self,other}_{vision,motion,position}
data/data/r3_stay_viz20/data.h5           同上
```
両方とも `other_vision` を含む（shape `(20, 101, 16, 64, 3)` uint8）。**新規収集は不要、そのまま流用可能。**

**(c) 既存作図スクリプトの流用可否**：
| スクリプト | 引数化 | v6での扱い |
|---|---|---|
| `plot_state.py` | `--result_dir`/`--epoch`/`--mode`等、汎用的 | そのまま使える見込み |
| `plot_pca_state_v4.py` | `--h5`/`--epoch`/`--label`、汎用的 | そのまま使える見込み |
| `regression_baseline_v4.py` | `--saved_h5`/`--exp_config`等、汎用的 | そのまま使える見込み |
| `plot_training_curve.py` | `--exp_config`（複数可）、汎用的 | そのまま使える見込み |
| `analyze_vpt.py` | `--epoch`/`--mode`等 | 要確認（Fig.5視点取得、v4 Encoder前提の箇所がないか未検証） |
| `r4_ve_eval.py` | `--exp_config`/`--epoch`/`--seed`、8次元Q̂²前提のcos_sim計算 | **8次元Q̂²の評価にはそのまま使える**（v6もSM入力は8次元のまま）。ただし方向計算 (`to_green`等) の対象がQ̂²のままか確認要 |
| （指示書記載の）`plot_q2hat_vs_trueq.py` | — | このファイル名は現存しない。近い役割の `analyze/plot_q2hat_v3.py` / `analyze/plot_q2hat_by_a2pos.py` が該当。**これらを `r̂²`(4次元) vs 真の報酬パラメータ用に改修する必要がある**（指示書の指摘通り） |

**結論**：図の材料はほぼ揃っている。新規に書く必要があるのは「`r̂²` vs 真の報酬パラメータ」散布図（S5用、論文の核心図）のみで、既存の `plot_q2hat_v3.py` 系を土台に改修すれば良い。

### 4.4 GPU・ディスクの空き状況（2026-09-12 時点、要再確認してから各段を起動）

**GPU**（`nvidia-smi`、8基 Quadro RTX 6000 24GB）：

| GPU | 使用メモリ | 使用率 | 状態 |
|---|---|---|---|
| 0 | 2049 MiB | 0% | 他ユーザ（matsuoka）が保持中（プロセス20日超） |
| 1 | 17468 MiB | 0% | 他ユーザ（matsuoka）が保持中 |
| 2 | 19924 MiB | 0% | 他ユーザ（matsuoka）が保持中 |
| **3** | **8 MiB** | **0%** | **空き** |
| 4 | 15674 MiB | 0% | 他ユーザ（matsuoka）が保持中 |
| 5 | 18078 MiB | **97%** | 他ユーザが実際に計算中（触らない） |
| 6 | 8802 MiB | 0% | 他ユーザ（matsuoka）が保持中 |
| **7** | **8 MiB** | **0%** | **空き** |

**GPU3・GPU7が現時点で空き。** v4ではGPU3/GPU5を使っていたが、GPU5は現在他ユーザが実計算中のため使用不可。**S1着手時にGPU7も候補に加え、都度 `nvidia-smi` で再確認すること**（他ユーザの状況は変動する）。

**ディスク**（`df -h`）：
```
/dev/sdb1  894G  842G  53G  95%  /home   ← /home/kusano/superposition はここ
/dev/sda3  876G  370G  507G  43%  /
```

**⚠️ 重要**：`/home` マウントは既に95%使用、**空き53GBのみ**。v4で100%まで埋めて夜間ジョブが全滅した事故と同じディスクである。指示書§6.2の見積もり（チェックポイント2〜3GB、評価中間saved.h5は1本3GBだが都度削除）自体は53GBに収まる規模だが、**空きに対する余裕が小さいことを認識して、通常以上に慎重な削除運用が必要**。長時間ジョブの前には必ず`df -h`を再確認し、必要量の1.5倍の空きがなければ起動しない（指示書の既定ルールを、通常より厳格に適用する）。

---

## 5. 未解決の懸念・要決定事項（S1着手前に固定すべきもの）

1. **報酬パラメータ`r`のサンプリング分布が未定義**。指示書は「エピソードごとにランダムにサンプル」としか書いていない。候補：(a) A-1/A-2の2点のみを交互に使う、(b) 4軸から2軸を選び`±1`を割り当てる12通りの離散サンプル、(c) 4次元球面上の連続一様分布。(b)は評価時の2条件（true_r/wrong_r）と対称的でcriticの汎化範囲を広く取れる一方、(c)は最も一般的だが「軸に沿った」構造をcriticが学習しやすいかは未検証。**S1着手前にこの一点を決定し、事前登録すること。**
2. **Actorも`r`で条件付けるべきか**。4.2で述べた通り、Actorが`r`を無視すると、SACのターゲットQ計算に使う方策がrごとに最適化されず、収束後のcriticの精度に影響しうる。実装コストは小さい（ActorLSTMのforwardにも`r`を足すだけ）ので、**やっておいた方が安全という判断に傾いているが、最終決定はS1着手前に行う**。
3. **cyan/blueのピクセルしきい値が未実測**。4.1参照。S1のreward計算実装前に実データでヒストグラムを取って確定させる。
4. `analyze_vpt.py`（Fig.5視点取得）がv4 Encoder前提のハードコードを含むかどうか未検証（他スクリプトほど深く見ていない）。S2/S3着手前に確認する。

## 6. 棄却された仮説

（まだなし。棄却が発生した都度、v4の付録と同じ形式で追記する：仮説 / 棄却の根拠）

## 7. 失敗した実験・撤回した判断

（まだなし）
