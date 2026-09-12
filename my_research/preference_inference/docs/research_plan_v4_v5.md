# 研究方針・実装指示書（v4 完走 → v5）

**作成**: 2026-09-03
**位置づけ**: 修論（M2、残り約2ヶ月で実装＋執筆）に向けた確定方針。
`v4_experiment_log.md` が実験の一次記録、本ファイルが「これから何をどの順でやるか」の指示書。
このファイルを Claude Code に渡してタスクを実行させる粒度で書く。

---

## 0. 認識合わせ（これで確定）

### 0.1 研究の問い（縮小・確定版）

> **元論文（Noguchi et al. 2022）の SM 入力である運動指令 `m_t` を行動価値 `Q(s,·)` に置き換えても、
> (a) A-1 の視覚予測、(b) A-2 の視覚予測、(c) A-2 の価値予測 はできるのか。**

- 「他者を価値主体とみなす枠組みが最小前提から創発するか」という大きい問いは**博士テーマ**。修論では上記に絞る。
- RL critic・報酬関数は**与えられた前提**（価値機構そのものの創発は主張しない）。単一環境・手設計報酬もスコープとして明記する。

### 0.2 2フェーズ構成

| フェーズ | 内容 | 問いのどこに答えるか |
|---|---|---|
| **v4 完走（P0）** | 評価を決定的にし（P0-0）、設計忠実性を検証し（P0-2）、3シードで固め、v4 を完走・評価しきる | (a) を確定、(b)/(c) の境界を**P0-2 の後に**「フリーズ Encoder 前提」付きで回答 |
| **v5（P1）** | Visual Encoder も含めて最初から Q 値入力で学習し直したベース段 | 「機構そのものが価値の下で自己組織化するか」＝問いの核心 |
| **P1.5** | 単流統制（process-2 を殺す）。「時間があれば」ではなく必須 | (b) の h²→other が機構由来か副産物か |
| **v5 拡張（P2、時間があれば）** | v5 ベース段に VE を載せて A-2 移動条件へ | (b) generative・(c) をフリーズ交絡なしで回答 |

### 0.3 スコープから外すもの（Future Work に明記して触らない）

R5（自己投影バイアス）／A-1 の緑嫌悪除去／A-2 の GAMMA 再学習／多選好の選好分類（＝交絡で撤回した旧・貢献2）／mg7 PCA スコア／R3-curriculum／v5 フルチェーン（v5 ベース→VE→R4-move まで通す完全版）／`h¹→other` 漏れ（0.585）の原因特定。

### 0.4 スケジュール（今日 9/3 起点、11月頭に初稿）

| 週 | 実験 | 執筆 |
|---|---|---|
| W1 (9/3–9) | **`--test_name` パッチ → P0-0（評価決定性）→ P0-1 ハーネス**／P0-2 R4 忠実性チェック起動／P0-3 3シード再走起動／**v5 run1 の 10ep 試走で壁時計計測** | 目的・貢献文・アウトライン、Methods 骨子 |
| W2 (9/10–16) | P0 完了・評価集計（二重報告）／**P1: v5 ベース段（MSE→L1）起動**／P1.5 単流統制 | Methods 初稿、Related Work |
| W3 (9/17–23) | v5 ベース段 評価／P1.5 評価／（余れば P2 を1本） | Results 初稿・図 v1 |
| W4 (9/24–30) | **実験ハード凍結**。全数値を表に | Results 確定 |
| W5–6 (10/1–14) | — | Discussion／Limitations／Future Work／Intro／全体初稿を指導教員へ |
| W7–9 (10/15–11/4) | — | 改稿2回・要旨・結論・清書・バッファ |

---

## 1. 現状の到達点（P0 が埋めるもの）

| 問い | 実験 | 現状 | P0 で確定させること |
|---|---|---|---|
| (a) A-1 視覚予測 | R3-A / R3-stay | h¹→self ≈ 0.88〜0.91（**seed 0 のみ、かつ seed0 は3シード中唯一相転移した「当たり」**） | 3シード**全部が収束領域に入ってから**平均・誤差棒／P0-0 で決定性 |
| (b) A-2 視覚予測（生成なし） | R3-A | h²→other 0.90–0.98（A-2 が A-1 視野に映るため。superposition 由来か視覚エンコード副産物かは**未証明**） | **P1.5 単流統制**で「process-2／superposition が寄与しているか」を決める |
| (b) A-2 視覚予測（生成的・exp3 相当） | **P0-4（主結果）**／P0-2（対照・非忠実系列） | **P0-4（元論文に忠実：静止base→移動生成）が失敗**（真Q が視覚予測を悪化 `+0.00441±0.00048` late-5 頑健、方向 cos_sim −0.80）。**P0-2 は効果が測定限界以下＝判定不能**（`−0.00344±0.00375` FRAGILE）。「条件で符号反転」ではなく「忠実な条件で明確に否定的・非忠実は判定不能」。⚠️ **P0-4 は run 未収束**（§7 の注記） | P0-4 を主に確定。原因は§4ter（対称サニティ）待ち |
| (c) A-2 価値予測 | 同上 | **P0-4 が主結果として失敗**。原因は (i) Q 非通約性 (ii) 位置の方が損失を下げるため、の**どちらか未確定** | **対称サニティ（§4ter、格上げ済み）**でこの2仮説を切り分けてから確定 |

**貢献1（R3-A / R3′ critic_swap）は P0-3（3シード・収束後）完了まで暫定。** 貢献2（機構の自己組織化＝P1）と貢献3（(b)generative・(c) の境界）も暫定。
**貢献3 の主結果は P0-4**（元論文の構造＝静止base→移動生成に対応する系列。P0-2 は base 段階から A-2 が動く非忠実な系列で対照・歴史的経緯として残す——数値が良い方を主結果にする条件選択は避ける）。
「Q が自己と通約不可能だから失敗」という断定は**対称サニティ（A-2 の報酬を A-1 と同一にして Q 値域を揃える、§P2-b）の結果が出るまで保留**。
base→VE 段で h²→self が上がる（P0-2: 0.10→0.82／P0-4: 0.06→~0.5）事実は、
「Q 非通約性」以外に「位置の方が視覚予測損失を下げるため Q 値域によらず位置が優先される」という対立仮説も支持しうる。
**ただし方向は両系列で一致するが、数値は P0-2（収束済）のみ確定——P0-4 の h²→self は最終5チェックポイントで 0.09〜0.53 と振れ（run 未収束、§7 注記・`results_log.md` §4⑥）、上昇幅を固定できない。**
対称サニティで切り分けるまでどちらとも確定しない。P0 はこれらを
**3シード＋決定的評価＋忠実な設計**で防御可能な形にする作業。

---

## 2. 実行環境・共通のお作法

### 2.1 コンテナと作業ディレクトリ

```bash
docker exec -it kusano_research bash
cd /work/my_research/preference_inference     # train.py / test.py はここから実行
```

- **学習（train.py / test.py）は Xvfb 不要**（h5 からバッチ読み込みのみ）。
- **データ収集のみ** OpenGL が必要：`Xvfb :99 -screen 0 1024x768x24 & sleep 1` ＋ `DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3`。
- 長時間ジョブは接続断で死なないように：
  ```bash
  setsid nohup python -u train.py --exp_config <name> --seed <S> --device cuda:<N> \
      > data/result/<name>/train_seed<S>.log 2>&1 < /dev/null &
  ```
  （PID1 に再親化されるので `docker exec` セッション終了・VPN 切断の影響を受けない）
- GPU：`--device cuda:N`。空き確認は `nvidia-smi`。他利用者のジョブが載っていない GPU を選ぶ（過去ログでは GPU3/5/7 が空きがち）。

### 2.2 学習の仕組み（train.py）

```bash
python -u train.py --exp_config <exp_yml_basename> --seed <S> --device cuda:<N>
```

- `--exp_config r3_stay` → `config/exp/r3_stay.yml` を読む。
- モデルクラスは exp yml の `model.name`、そのハイパラは `config/model/<model.name>/<model.config>.yml`。
- 出力先：`data/result/<exp_config>/<seed>/`（`model/00010.pth` … `log/{train,eval,test}/*.log`）。
- **pretrain**：exp yml の `train.pretrain` に `{exp_config, seed, epoch}` を書くと
  `data/result/<pretrain.exp_config>/<seed>/model/<epoch:05d>.pth` を読んで初期値にする。
  `pretrain: ~` なら**スクラッチ**。
  形が合わない重み（SM の入力次元が m_t の 2 → probe-Q の K=8 に変わる等）は `util.load_pretrain` が
  自動でスキップ or 部分転移（`weight_ih` の視覚 64 列のみ移送、残りランダム）する。
- **freeze**：`train.optim_config.freeze` は**部分文字列リスト**。パラメータ名にどれかを含むと `requires_grad=False`。
  例：`- superposition_module` は `superposition_module.lstm.weight_ih` 等すべてを凍結。
  **`probe_critic` は必ず freeze に入れる**（A-1 の SAC critic は所与。入れ忘れると `gen_optim_params` が学習対象にしてしまう）。

### 2.3 評価用 saved.h5 の作り方（test.py）

```bash
python -u test.py --exp_config <name> --seed <S> --device cuda:<N> \
    --test_epoch <E> --test_data_name <dataset> --test_batch_size 10 \
    --test_modes eval \
    --save_targets self_position other_position state q2_hat a2_target_landmark
```

→ `data/result/<name>/<seed>/test/<test_name>/save/saved.h5` に
`<E:05d>/eval/state/self/hidden`（h¹）, `.../other/hidden`（h²）, `.../q2_hat`, position 系が入る。
`--mask_off` で `p_mask_vision=0` にできる（経路積分チェック用）。

**必須パッチ（他の全タスクの前・P0-0 手順2 より前に当てる）**：現状 `test.py` は保存先ディレクトリ名
`test_name` を `--test_data_name` と同一にしている（`util.gen_dirs(args, test=True, test_name=args.test_data_name)`）。
このため同じ (exp_config, dataset) で `test.py` を再実行すると既存の `test/<dataset>/save/saved.h5` を**上書き**する
（§2.5 の保護 dir 内でも起きる）。`test.py` / `args_util.add_test_args` に **`--test_name`（省略時は `--test_data_name` に一致）**
を追加し、`util.gen_dirs(..., test_name=args.test_name)` にする。これでハーネスは
`--test_data_name r2_a1random_a2rl --test_name r2_a1random_a2rl__evalv4v5_maskoff` のように
**評価設定ごとに別サブdir**へ書ける。既存の `test/<dataset>/` には触れない。
- 変更は `test.py` と `args_util.py` の2ファイルのみ、後方互換（`--test_name` 未指定なら従来通り）。
- **このパッチが当たるまで、既存 config への `test.py` 実行（P0-0 手順2 含む）を一切行わない。**

### 2.4 記録の義務（毎実験）

各結果 JSON に **git commit hash ＋ seed ＋ dataset ＋ config 全文** を必ず含める
（`util.gen_result_metadata(exp_config_name=..., seed=..., dataset_name=...)` を merge する。
既存 analyze スクリプトはこれを呼んでいる。新規スクリプトも必ず呼ぶ）。

**加えて `docs/results_log.md` に追記する**（追記専用・行を消さない）：
- 学習を起動したら §1 run 登録簿に1行（exp_config / seed / GPU / epochs / 目的 / pretrain / 状態 / 出力パス）。状態が変わったらその行を更新。
- 評価（P0-1 ハーネス or 個別 analyze）を回したら §2 の該当表に1行（seed / epoch / 評価データ / eval_seed・マスク / 数値 / commit / 日付 / file）。
- `v4_experiment_log.md` は「なぜ」（設計・棄却仮説・解釈）、`results_log.md` は「何を回して何が出たか」だけ。混ぜない。

### 2.5 絶対に触らない（削除・上書き禁止）

- root 側（`/work/` 直下、`my_research/` 以外）＝元論文の実装すべて。config も analyze もコピーして使う。
- `/work/data/result/{exp1_l1,exp1_l1_1000,exp2,exp3,exp3_1000}/`、`/work/data/data/{grid,self_stay_other_random,self_random_other_stay*}/`
- `data/result/baseline_v4/`（v4 の全測定結果。追記のみ可）
- `data/result/{r3_a_direct,r3_a_direct_1000pretrain,r3_a_direct_1000pretrain_400,r3_stay,r3_stay_400,r4_ve}/`（既存の学習済み。新規 run は**別 exp_config 名**で）
- `data/data/{r3_stay,r2_a1random_a2rl}/data.h5`（読み取りのみ）
- `my_research/step2/`, `my_research/step3/`（別研究ライン）

**注記1（新規 seed 追加は可）**：`util.gen_result_dir` は `data/result/<exp_config>/<seed>/` で、seed ごとに
ディレクトリが分かれる（コードで確認済み）。既存 `data/result/<exp_config>/0/` は不変のまま、同一 exp_config へ
**新しい seed（1, 2）を追加するのは許可**。P0-3 はこれに該当する。既存 `<seed>/` サブツリーへの上書きは禁止。

**注記2（`test.py` の上書き対策）**：§2.3 の `--test_name` 分離パッチを**必ず先に**当てる。パッチ後は
`test/<dataset>/save/` ではなく `test/<test_name>/save/` に書くので、評価設定ごとに別サブdirへ隔離でき、
保護 dir 内の既存 `test/<dataset>/` を破壊しない。パッチ前に既存 config へ `test.py` を実行しない。

### 2.6 実行前の書き込み先申告（各タスク必須）

どのタスクを実行する場合も、着手前に **(1) 生成・編集・上書きするパス全部を列挙 → (2) §2.5 の禁止リストと1件ずつ照合 →
(3) ⚠️（保護 dir 内・既存ファイル上書き）があれば回避策と共に報告** する。実行後は `git status` で
「my_research 配下の新規 or 意図した変更のみ」を確認。root 側・削除禁止資産に差分が出ていたら**即座に停止して報告**。

本ファイル作成時点の全タスクの書き込み先一覧（照合済み）は §9 付録。新タスク追加時はそこに追記。

---

## 3. P0 — v4 の完走と地固め（最優先・全部やる）

> **P0 内の順序は固定**：`--test_name` パッチ（§2.3） → P0-0 → P0-1（受け入れ確認）→ P0-2 / P0-3 / P0-4 / P0-5。
> **`--test_name` パッチが P0-0 手順2 より前**（逆だと最初の決定性チェックで既存 `saved.h5` を潰す）。
> **P0-0 が完了するまで P0-3（3シード再走）と P0-4 の評価に着手しない。** 評価が非決定的なままだと
> シード間ばらつきと評価ノイズを区別できず、3シードにする意味がなくなる。

### P0-0. 評価の再現性を確保する ← **最優先。他の全 P0 タスクの前提**

**判明している問題**（直近の調査）：
- `util.mask()` は `torch.rand(...) < p_mask` で毎回サンプリングする（`model/util.py`）。`p_mask_vision=0.99` のランダムマスクが
  **評価時にも効いている**。`test.py` は冒頭で `util.seed_all(args.seed)` を呼ぶので「新規プロセスでの最初の eval」は
  シード固定なら決定的だが、(a) 同一プロセス内で eval を2回（`--test_modes eval test`）、(b) `r4_ve_eval.py` 等が独自に forward を
  回す、(c) `--seed` を毎回渡していない、のいずれかで乱数列がずれる。
- 実測：同一チェックポイントを2回評価すると `feature_prediction_self` が **14.86 → 14.79** と変動。
- 実測：R4-move は ep130/270/400 で結論が反転（Q̂²→A-1位置 R² = **0.315 / 0.659 / 0.006**、`real` vs `zero` の大小も、
  方向 cos_sim の符号も反転）。これは評価ノイズ ＋ 学習の「停滞→相転移／振動」（`v4_experiment_log.md` §7.12, §8.4）の合わせ技。

**やること**：
0. **（前提）`test.py` の `--test_name` 分離パッチを当てる**（§2.3）。手順2 でこれが無いと決定性チェックの再実行が
   既存 `saved.h5` を上書きしてしまう。パッチ → 後方互換確認（`--test_name` 未指定で従来と同じパスになること）→ 以降へ。
1. **評価経路のシード固定**：`test.py` と全 analyze スクリプト（`regression_baseline_v4.py` / `r4_ve_eval.py` /
   `probe_q_direct_regression.py` / `q2_position_regression.py`）で、モデル forward を回す**直前**に
   `random.seed(S); np.random.seed(S); torch.manual_seed(S); torch.cuda.manual_seed_all(S)` を明示的に呼ぶ。
   使用シード `S` は固定値（例 0）に統一し、CLI 引数 `--eval_seed` として受けられるようにする。
2. **決定性の確認**：同一チェックポイント・同一 `--eval_seed` で `test.py` を2回走らせ、`saved.h5` の
   `state/self/hidden` と `state/other/hidden` が **bit-identical**（`numpy.array_equal`）になることを確認。
   ならなければ乱数源が残っている（`torch.backends.cudnn.deterministic=True` も設定、`--cudnn_deterministic` フラグ）。
3. **メタデータ**：全結果 JSON に `eval_seed` と「マスク on/off」を記録（`gen_result_metadata` に項目追加）。
4. **元論文の作法を確認**：root `exp/runner.py` / `test.py` / `run_analysis*.sh` で、
   - 評価時にマスクを掛けているか（掛けているなら p=いくつか）
   - eval と train で分岐やシード固定をしているか
   を読んで、v4 の評価をそれに揃えるべきか判断。**判断結果を本ファイルと `v4_experiment_log.md` に記録**。
5. **canonical eval の決定**：以下の2案から選ぶ（4 の結果を踏まえて）。決めたら全条件で統一：
   - (A) `--mask_off`（p=0、マスクなし）→ 構成的に決定的。ただし訓練時と条件が違う
   - (B) マスク on（p=0.99）＋ `--eval_seed` 固定 ＋ **N=5 マスク実現でmean±sd**（評価ノイズ自体を数値化して報告）

**受け入れ基準**：手順2が bit-identical。これが通るまで P0-1 の受け入れ判定（既知数値の再現）は保留。

---

#### P0-0 実施記録（2026-09-04）

- **手順0（`--test_name`/`--eval_seed` パッチ）**：✅ 完了（`args_util.py`, `test.py`。後方互換確認済）。
- **手順1（評価経路のシード固定）**：✅ 完了。
  - `test.py`：`--eval_seed` 指定時、eval ブロック直前で `util.seed_all` ＋ `cudnn.deterministic=True`。
  - `analyze/r4_ve_eval.py` / `analyze/q2_position_regression.py` / `analyze/probe_q_direct_regression.py`：
    `--eval_seed`（default 0）を追加、`main()` 冒頭で `random`/`np.random`/`torch`/`torch.cuda` を全 re-seed ＋ cudnn deterministic。
    結果 JSON の `eval_seed` もハードコード 0 → `args.eval_seed` に。
  - `analyze/regression_baseline_v4.py`：randomness なし（Ridge を全データに fit するだけ）→ 変更不要。
- **手順2（決定性確認）**：✅ **通過**。`test.py --exp_config r3_a_direct_1000pretrain --seed 1 --test_epoch 200
  --test_data_name r2_a1random_a2rl_viz20 --eval_seed 0 --cudnn_deterministic --save_targets state` を
  `--test_name determ_a` / `determ_b` で2回実行 → `state/self/hidden` `state/other/hidden` が **`array_equal=True`, max|A-B|=0**。
  （p=0.99 のマスク乱数を含む `PredictionFeaturePredictionRunner` モデルで確認）。
- **手順4（元論文の作法）**：root `test.py` は冒頭で `util.seed_all(args.seed)` を呼ぶ（eval もシードされる）。
  `run_analysis.sh` / `run_analysis_exp2.sh` は常に `--seed 0`、`--mask_off` は**渡さない** → **元論文の canonical eval ＝
  マスク on（config の `p_mask_vision`）＋ seed 0**。元論文は eval ブロック直前で re-seed はしない（`--eval_seed` はその強化版）。
  root は `--test_modes eval test` を既定で走らせる＝eval が RNG を消費した後に test が走る（旧 v4 の「同一 ckpt で数値が違う」は
  おそらく train split(`eval` mode) vs test split(`test` mode) の混同、または invocation 差。真の非決定性ではなかった可能性が高い）。
- **手順5（canonical eval）**：✅ **ユーザー承認済（2026-09-04）**。
  - **主指標** ＝ マスク on（config `p_mask_vision`）＋ `--eval_seed 0` ＋ `--cudnn_deterministic`
    （元論文の「seed＋マスク on」と一致、手順2 で bit-identical 実証済）。
  - **split** ＝ 上記「4軸 R² の報告」①canonical（`r2` の `eval` グループ＝訓練split）を主、②in-distribution・③held-out を併記。
  - **主要条件**（`r4_ve_exp3parity` / R3-A 3seed / R3-stay）で `--eval_seed 1..4` も回し、
    **マスク実現の sd（評価ノイズ）** を **訓練 seed の sd** と分けて報告。
    → **報告時に「n=5 の粗い推定」と明記する**（査読で突かれる点。ユーザー指示 2026-09-04）。
  - `--mask_off` は補助サニティとして各条件1回のみ（主にしない）。

---

### P0-1. 統一評価ハーネスを作る

**目的**：条件間で比較可能にする。いまは on/off-policy・データセット・epoch がバラバラ（`v4_experiment_log.md` §7.10, §10）。

**作るもの**：`analyze/run_eval_v4v5.sh <exp_config> <epoch> <seed>`（新規、既存 analyze スクリプトを呼ぶだけのラッパ）

**前提**：§2.3 の `--test_name` パッチ済み ＋ P0-0（評価の決定性）が通っていること。

#### split の意味（コードで確認済・2026-09-04）

`util.gen_data_loader`：`eval_loader = DataLoader(data['train'], shuffle=False)`、`test_loader = DataLoader(data['test'])`。
つまり **`test.py --test_modes eval` は「訓練split・unshuffled」を読む**（held-out ではない）。`--test_modes test` が held-out（`r2`/`r3_stay` は各 300ep）。
**全 v4 の 4軸 R² 数値（`regression_baseline_v4.py --mode eval`）は訓練split で測られている**（JSON で確認：`exp1_l1_1000_r2.json` は `mode:"eval"`, `n_data:1000`, dataset `self_random_other_stay_1000`）。
これは**元論文 Fig.4d の回帰も同じ**（`.../save/regression/eval/plot/` ＝ 同じ `eval` グループ ＝ 訓練split）。frozen モデルの隠れ状態から位置を線形デコードする**表現プローブ**なので、訓練データ上で測ることは元論文と一貫しており leakage 懸念は通常の意味では小さい。**ただし「訓練split 上の線形デコード可能性」と明記する。**

#### 4軸 R² の報告（3種、いずれも `regression_baseline_v4.py`）

| # | データセット | split | 目的 |
|---|---|---|---|
| ① **canonical** | `r2_a1random_a2rl`（A-2 移動） | `eval` グループ（訓練split・unshuffled） | **条件間の横並び比較専用**。全 v4 既存値・元論文 Fig.4d と同じ方法。静止訓練モデル（R3-stay / v5-base）もこれで測る＝A-2 移動条件への汎化を見る（OOD だが全条件同一なので比較可） |
| ② in-distribution | 各モデルの訓練データ（R3-stay/v5-base→`r3_stay`、R3-A/R4-*→`r2_a1random_a2rl`、元論文 exp1_l1_1000→`self_random_other_stay_1000`） | `eval` グループ | **元論文ベースライン（0.9667 / 0.0613 / 0.0472 / 0.9932）との直接比較専用**。②の exp1_l1_1000 値がまさにそれ。※データセット・エピソード数が条件で違うので①の代わりにはならない |
| ③ held-out ロバスト性チェック | ①②と同じデータセット | `test` グループ（`--test_modes test`、300ep） | 訓練split で測った①②が汎化するかの確認。**主要条件のみ**（exp1_l1_1000 / R3-A 3seed / R3-stay / P0-2 / P0-4）。①②と乖離しなければ表現プローブは頑健 |

表では①（条件間比較）②（元論文比較）③（held-out）を**列で分ける**。混同しない。

固定プロトコル（両報告で共通）:
- `--test_epoch` = 各 run の**最終**エポック（P0-5 の学習曲線で収束を確認してから。R4-move は特に P0-0 で見た
  「複数チェックポイントで結論反転」があるので、収束基準を満たす epoch を1つ選び、その1点のみ報告）
- マスク = P0-0 手順5 で決めた canonical 設定を全条件で統一
- `n_episodes` 固定（例 3000）、`--eval_seed` 固定
- `--test_name` は評価設定を含む一意な名前（例 `r2_evalv4v5` / `indist_evalv4v5`）

ハーネスがやること（順に）:
1. `test.py ... --save_targets self_position other_position state q2_hat a2_target_landmark --test_modes eval` で saved.h5 生成
2. `python analyze/regression_baseline_v4.py --saved_h5 <saved.h5> --epoch <E> --mode eval --label <exp_config>_seed<S> --exp_config <exp_config> --seed <S> --dataset_name r2_a1random_a2rl` → 4軸 R² JSON
3. `python analyze/probe_q_direct_regression.py --exp_config <exp_config> --epoch <E>` → 単一時刻 Q→位置（R3 系のみ）
4. VE 付きモデルなら `python analyze/r4_ve_eval.py --exp_config <exp_config> --epoch <E>` → real/zero/constant_mean/**true_q2** の 4条件視覚損失 ＋ Q̂² 統計 ＋ 方向 cos_sim
5. VE 付きモデルなら `python analyze/q2_position_regression.py --exp_config <exp_config> --epoch <E>` → Q̂²→A-1位置 vs →A-2位置 R²
6. `python analyze/plot_training_curve.py --exp_config <exp_config>` → 学習曲線 PNG

**成果物**：`data/result/baseline_v4/<exp_config>_seed<S>_{r2,r4ve,q2pos}.json` ＋ 曲線 PNG。全部 `gen_result_metadata` 付き。

**受け入れ基準**：既存の `r3_a_direct` / `r4_ve` に対してハーネスを回し、`v4_experiment_log.md` §5.1・§5.7 の既知数値を再現できること（再現できなければハーネスのバグ）。

---

### P0-2. R4 の設計忠実性チェック ← **重要。ユーザーの「設計が正しいか不安」に直接答える**

**発見済みの不一致**（コード実測）：元論文 exp3（MG）と v4 の R4（VE）で、runner が違う。

| | 元論文 exp3 / v4 R2 | v4 R4-ve / R4-move |
|---|---|---|
| runner | `MotionGenerationPredictionFeaturePredictionRunner` | `PredictionFeaturePredictionRunner` |
| t>0 の視覚マスク | **1.0（完全マスク）** | `p_mask_vision` = **0.99** |
| FPM 特徴の detach | **しない**（`predict_feature(ss)` / `predict_feature(os)`）→ `L_feat_other → os → SM → 生成モジュール` に勾配が流れる | **する**（`.detach()`）→ **`L_feat_other` から VE に勾配が流れない** |

**含意**：v4 の VE は、元論文の MG が受けていた「FPM_other → 隠れ状態 → 生成モジュール」の学習信号を**受けていなかった**。
加えて t>0 マスクが 1.0 でなく 0.99 ＝ 視覚が 1% 漏れる楽な代替経路がある。**「弱い信号 ＋ 楽な抜け道」**の状態。
Q̂² が A-1 状態を符号化したことも、チェックポイント間で振動したことも、これで説明できる可能性がある。
これは Q 値の非通約性（`v4_experiment_log.md` §7.9）とは**独立した、R4 失敗の第一容疑**。

> **runner 名について（誤解注意）**：`MotionGenerationPredictionFeaturePredictionRunner` は
> 実体が「完全マスク（t>0 で 1.0）＋非 detach の FPM 損失」の2挙動だけで、**MotionGenerator を
> instantiate も参照もしない**（生成モジュールが MG でも VE でも同じ）。元論文が exp3 で使ったので
> こう名付けられた歴史的な名前。本研究は VE を訓練するので、**P0-2 完了後の `exp/runner.py` バッチで
> 同一コードのエイリアス `GeneratorTrainingFeaturePredictionRunner` を追加**し、P0-4 / v5 の config は
> そちらを指す。P0-2 は挙動がクラス名に依存しない（出力 bit 同一）ため再起動しない。以下「exp3-parity runner」
> と書いたら `MotionGeneration...`（現状）＝`GeneratorTraining...`（改名後）を指す。

**方針（分岐させない）**：exp3-parity（非 detach・完全マスク）が**元論文に忠実な設計**なので、
**新規 run はすべて最初から exp3-parity runner で回す**。改善しようがしまいが parity を使う——
「元論文と違う設計で失敗しました」では境界主張が弱い。忠実な設計で失敗して初めて「機構の限界」と言える。
既存の非 parity 版（`r4_ve` / `r4_move`——`r4_move` は 2026-08-24 に 400ep 完走・評価済）は**歴史的経緯・対照として残す**（消さない）。

**やること**：
1. `config/exp/r4_ve_exp3parity.yml` を新規作成：`r4_ve.yml` のコピーで **`runner:` 行だけ**
   `MotionGenerationPredictionFeaturePredictionRunner` に変える（このランナーは `gen_mask_prob` が t>0 で 1 を返すので `p_mask_vision` は無視される＝完全マスクになる）。
2. `SuperpositionNetworkProbeQValueEstimation` がこのランナーで動くか**スモークテスト**（forward 署名・`init_state`・`get_state()['self'/'other']`・`predict_feature` は互換のはず。1エポックだけ回して落ちないか確認）。
3. `python -u train.py --exp_config r4_ve_exp3parity --seed 0 --device cuda:<N>`（200 epoch、pretrain は `r3_a_direct_1000pretrain` のまま）。
4. P0-1 ハーネスで評価し、非 parity `r4_ve`（既存）と横並び。

**判定（parity 版の結果を正とする）**：
- parity 版でも VE が機能しない（Q̂²→A-2位置 R² が低いまま、`true_q2` ≥ `zero`）→ runner の不一致は主因でない。
  **忠実な設計でも失敗した ＝ Q 非通約性（貢献3）が確定的に主張できる。**
- parity 版で改善（Q̂²→A-2位置 が上がる／`true_q2` < `zero` に逆転）→ **旧 R4 の失敗は一部ビルドミスだった。**
  `v4_experiment_log.md` の §5.7・§7.9 と本ファイル §7 を書き換え。以降 (c) の主張は parity 版に基づく。

**r4_move も parity で回し直す**（P0-4、`r4_move_exp3parity` 新規）。既存の非 parity `r4_move`（2026-08-24 完走・評価済、**未収束**＝`feature_prediction_other` が 35→68 に発散、eval も ep で結論反転）は**対照として残す**。

---

### P0-3. R3-A / R3-stay の 3シード再走

**理由**：`v4_experiment_log.md` §5.1 の全数値が seed 0 のみ。誤差棒なしで「h¹→other 0.585」「R3-stay で 0.41→0.54 悪化」等を主張できない。

**やること**（seed 1, 2 を追加。seed 0 は既存を流用）。**GPU0 のみ・順次実行**（seed 並列は計算競合で両方遅くなる。
P0-3 は critical path でない＝P0-2 の後に出れば良いので、確実な順次を選ぶ。GPU4/5/6 は他ユーザー確保のため使わない）:
```bash
# seed1 → seed2 を1本のチェーンで（setsid nohup bash -c "python ... seed 1 ; python ... seed 2" &）
python -u train.py --exp_config r3_a_direct_1000pretrain --seed 1 --device cuda:0
python -u train.py --exp_config r3_a_direct_1000pretrain --seed 2 --device cuda:0
# r3_stay_400 の seed1,2 は GPU0 に空きが出てから（同じく順次）
python -u train.py --exp_config r3_stay_400 --seed 1 --device cuda:0
python -u train.py --exp_config r3_stay_400 --seed 2 --device cuda:0
```
- config はそのまま（`--seed` だけ変える。`util.seed_all` がシード適用）。
- 出力は `data/result/<exp_config>/<seed>/` に分かれる（`gen_result_dir` がコード確認済み・§2.5 注記1）。既存 seed 0 は不変。
- **注意**：config の `train.pretrain.seed` は 0 に固定されているので、seed 1,2 も pretrain 元は **seed-0 の
  `exp1_l1_1000/0`**（`self_random_other_stay_1000` で学習した凍結エンコーダ／デコーダ）。つまり 3 シードで変わるのは
  **SM の初期化 ＋ データ順 ＋ マスク実現**のみで、視覚フロントエンドは共通。「誤差棒は SM 学習部分についてのもの」と
  Limitation に明記する（フルの seed 研究ではない）。
- 完了後、各 seed に P0-1 ハーネス → `regression_baseline_v4.py` の 4軸 R² を seed ごとに JSON 化 → **mean ± sd の表**を `data/result/baseline_v4/r3_multiseed_summary.json` に。

**受け入れ基準**：seed 0 の既知値が mean ± sd の範囲に入る。sd が大きすぎる（例 h¹→other で ±0.15 超）なら、その指標は「条件間の差」を主張しない。
P0-0 の評価ノイズ（同一チェックポイントの再評価ばらつき）と、この seed 間 sd を**分けて報告**する。

---

### P0-4. R4-move の完走と評価

- 既存の `r4_move`（非 parity、pretrain = `r3_stay_400` ep400、**2026-08-24 に 400ep 完走・評価済**）は **対照として保持**（消さない）。未収束（`fp_other` 発散）・eval が ep130/270/400 で反転、が P0-0/P0-5/parity runner を要する実例。
- **主とするのは parity 版**：`config/exp/r4_move_exp3parity.yml`（`r4_move.yml` のコピーで `runner:` を
  `MotionGenerationPredictionFeaturePredictionRunner` に）。P0-2 のスモークテスト通過後に起動（400 epoch）。
- 評価は P0-1 ハーネス（parity 版・非 parity 版の両方に適用して並べる）。**主指標**：
  1. `r4_ve_eval.py` の **`true_q2` vs `zero` vs `real`**（`true_q2 ≥ zero` なら「A-2 の正解を入れても得しない」＝(c) 不成立）
  2. `q2_position_regression.py` の **Q̂²→A-1位置 R² vs →A-2位置 R²**（非 parity では 0.43 vs 0.08。逆転するか）
  3. Q̂² の std（崩壊していないか）、方向 cos_sim（緑を向くか）
- **epoch の選び方**：P0-5 の学習曲線で収束（相転移後の安定領域）を確認 → その region から **1 epoch を選び、その1点のみ**を
  P0-1 で評価。P0-0 で見た「ep130/270/400 で結論反転」を繰り返さないため、複数 epoch の数字を並べて選ばない。

---

### P0-5. 収束判定（全 run 必須・閾値ベース）— **確定（2026-09-06、ユーザー承認）**

`v4_experiment_log.md` §7.12：Q 値入力モデルは「長い停滞 → 急激な相転移」型で、最終エポックだけ見ると学習不足を「性能が悪い」と誤診する。
かつ相転移は**シード依存**（R3-A：seed0 は ep170、seed1 は ep~230、r3_stay：seed1 は 400ep でも未相転移）。
→ 目視でなく**閾値で判定**し、結果を2段構えで報告する。

**判定対象**：`feature_prediction_other`（train ログ、毎エポック）。E=総エポック、W=最終20%（≥30ep）、B=バーンイン後の全区間（ep ≥ ⌈0.1E⌉）。

**「収束」⇔ 以下の①②を両方満たす**（③相対基準は「後から seed 追加で既存判定が変わる／全高止まりで全部通る」問題があり**落とした**、`min(B)` の絶対値を別途報告して代替）：
1. **水準**：`mean(fp_other over W) ≤ 1.25 × min(fp_other over B)`（自分の最良値付近で落ち着いた）
2. **後半の非上昇**：`|OLS slope over W| × |W| ≤ 0.10 × min(fp_other over B)`（最終窓の正味ドリフト < 最良値の10%）

**per seed で記録**：`mean(W)` / `min(B)` / `slope(W)×|W|` / 相転移エポック（`fp_other < 1.3×min(B)` の初回、無ければ「なし」）/ ①②の pass・fail。
**`min(B)` の絶対値**が「相転移せず高止まり」の判別材料（例：r3_stay 収束シードは ~30、未相転移シードは ~54 で、①は通っても `min(B)` の値で一目瞭然）。

**報告（2段構え）**：
- (1) **収束した seed のみ**で 4軸 R² の mean±sd（n を明記、**n≤2 なら sd は「参考値」**と付記）
- (2) **収束率** =（収束 seed 数）/ N を別途報告 ＝「Q値入力は収束が不安定・シード依存」の直接証拠（§7.12 を3シードで定量化）

判定スクリプト：`analyze/check_convergence.py`（新規）。曲線 PNG は補助（`plot_training_curve.py`）。

#### 評価点の選び方（2026-09-06 確定、ユーザー指示）：**最終5チェックポイント平均を全 run に一律適用**

- P0-1 の全評価（4軸R²・`true_q2` アブレーション・`Q̂²→位置`・cos_sim）は、**最後の5チェックポイント（E, E−10, …, E−40）** で
  それぞれ実施し、**平均±sd** を報告する。
- **位置で選ぶ（最後の5個）**——訓練損失が最小の epoch を選ぶ (a) は採用しない（訓練損失最小 ≠ 評価最良、かつ「都合の良い点」批判を招く）。
- **主結果だけでなく全 run に同じ処理**（P0-2 / P0-4 / R3-A_400 3seed / R3-stay_400 3seed / 元論文 / v5 …）。
- **チェックポイント間 sd を必ず併記**——これは `--eval_seed` の sd（マスク評価ノイズ）とは**別物**で、学習の振動が結果に与える影響。
  判定基準：**`|mean(true_q2 − zero)| > チェックポイント間 sd` なら主張が生き残る。sd の方が大きければ主張は成立しない。**
  （P0-2 の −0.0011 は元々小さいので特に危うい。P0-4 の +0.0046 も要確認。）
- スクリプト：`analyze/run_eval_lateckpt.sh <exp> <E> [seed] [dataset]`（ハーネスを5 epoch 分ループ）＋ `analyze/aggregate_lateckpt.py`（mean±sd 集計、上記判定を出力）。

---

### P0-6. 視覚出力の保存（論文図用）— 学習中に設計だけ確定、取得は評価時

**なぜ今やるか**：`--save_targets` の既定（`self_position other_position state q2_hat`）に視覚が無い。
「視覚予測ができた」と主張する以上、数値だけでなく元論文 Fig.4c / Fig.5c 相当の**実画像**が要る。
3シード・v5 の結果が出てから「画像が無い」と気づくと取り直しになる（コード実測済みの制約は以下）。

---

#### 確定図リスト（2026-09-06 固定。**生成は後回し**＝学習不要・再生成可能なので急がない。ここでは「何の図を・どの条件で・何を主張するために・いくらで」だけを確定する）

凡例：**必須**＝論文本文の主張に直結（無いと結論が書けない）／**任意**＝補足・診断（査読対策や付録）。コスト欄の「作図のみ」＝追加の学習も評価実行も不要、既存ファイルからスクリプトを回すだけ。

| # | 図種 | 条件 | 必須/任意 | この図で何を主張するか（1行） | コスト・材料 |
|---|---|---|---|---|---|
| F1 | 学習曲線（`feature_prediction_self` / `feature_prediction_other` / total loss vs epoch） | 全 run（元論文 exp1_l1_1000・exp3_1000／R3-A 3seed／R3-stay 3seed／P0-2／P0-4／非parity r4_move／v5 mse+l1／P1.75） | **必須** | 各 run が収束したか、価値入力系で「長プラトー→相転移」が起きるか・そのepoch・seed依存 | 作図のみ（train ログのみ）。`plot_training_curve.py` 既存 |
| F2 | 4軸 R² バー（h¹→self / h¹→other / h²→self / h²→other、①共通・②in-dist・③held-out の3種） | 表現プローブ全条件（exp1_l1_1000／R3-A／R3-stay／P0-2／P0-4／v5） | **必須** | h¹ は自己視覚を・h² は他者視覚を線形デコードでき、superposition（役割分離）が保たれているか | 作図のみ。`regression_baseline_v4.py`、late-5 の JSON 済（P0-2/P0-4）。R3-A/R3-stay/v5 は §P0-3/§P1 の eval で生成 |
| F3 | Fig.4c 相当 PCA state マップ（h¹・h² の hidden を2D PCA、A-1/A-2 の位置で色付け） | P0-4（主）／R3-A／R3-stay／v5、参照として exp1_l1_1000 | 主要条件**必須**・他**任意** | h¹/h² が空間的に構造化された表現を持ち、他者状態が自己と分離して符号化されているか（Fig.4c の定性版） | **作図のみ**（P0-2/P0-4 の saved.h5 に `state` 保存済）。R3-A/R3-stay/v5 は `state` 付き test.py 1回（軽・学習不要）。新規 `plot_pca_state_v4.py` |
| F4 | Fig.6b 相当 散布図（VE 出力 Q̂² vs A-2 の真 Q、相関 r） | **P0-4（主）／P0-2（対照）／P1.75（対称サニティ）の3枚並べ**＋元論文 Fig.6b（MG 生成運動 vs 実運動、r≈0.87）を参照 | **必須** | VE は A-2 の真の価値を回復できるか。元論文 MG は運動を回復できた（r=0.87）。P0-4 で失敗し、P1.75 で Q レンジを構造的に揃えても失敗するなら、原因は「Q 非通約性」ではなく「位置優位」 | **作図のみ**（P0-2/P0-4：saved.h5 の `q2_hat` ＋ データ h5 の `other_vision` ＋ A-2 critic）。P1.75 は学習後に `q2_hat` 保存の eval 1回。新規 `plot_q2hat_vs_trueq.py`（= `r4_ve_eval.py` の集計部＋散布図） |
| F5 | `true_q2` アブレーション バー（real / zero / constant_mean / true_q2 の self_vision L1、**相対%表示**、late-5 の inter-ckpt sd 付き） | P0-2／P0-4／P1.75 | **必須** | VE 出力を「A-2 の真 Q」に差し替えると視覚予測が改善するか悪化するか＝真 Q が SM にとって有用な入力か。late-5 で頑健性も示す | 作図のみ。`r4_ve_eval.py` ＋ `aggregate_lateckpt.py` の JSON 済（P0-2/P0-4）。P1.75 は学習後 |
| F6 | 方向ヒートマップ（Q̂² の最良プローブ方向 vs A-2 位置の空間マップ） | P0-2／P0-4／P1.75 | 任意（診断） | VE 出力の方向成分が A-2 位置に対し空間的に意味のあるパターンを持つか（絶対スケール非依存の弱いチェック） | 作図のみ。`r4_ve_eval.py` 既存出力 |
| F7 | 予測画像パネル（`self_vision`：input v_t／truth v_{t+1}／prediction v̂_{t+1}、＋ `other_vision` truth を横並び） | 元論文 exp1_l1_1000・exp3_1000／P0-4／R3-stay／v5 | **必須** | 数値だけでなく実画像で視覚予測の質を示す（A-1 予測がもっともらしいか、A-2 実際とどれだけ違うか） | **要・軽実行**：`other_vision` loader パッチ（下記「やること」手順1・P0-2後バッチ・**未適用**）＋ viz20 で `test.py --viz` 1回/条件（学習不要・≈0.4 GB/条件） |
| F8 | Fig.5 相当 VPT 再構成（Autoencoder で h² が「見ているもの」を復元、other_other / other_self 誤差＋再構成パネル） | 既存 fig5_v4_exp1l1000／fig5_v4_r3stay400 ＋ 新規 fig5_v4_r3a_direct／fig5_v4_r4ve_exp3parity／fig5_v5_base | (b) を強く主張するなら**必須**、他**任意** | h² から A-2 視点の画像が復元でき、それが「A-1 視点のコピー」ではない（§6 項目8 のアーティファクト回避） | 各条件 Autoencoder 100ep（fig5_v5_base のみ ≈2h、他は既存 or 軽）。`analyze/analyze_vpt.py` ＋ `build_fig5_v4_report.py` |
| F9 | 収束コントラスト図（`feature_prediction_other` 曲線の重ね描き：運動入力＝ep≤6 で収束 vs 価値入力＝長プラトー→相転移） | exp3_1000 vs R3-A vs r3_stay_400 | 任意（§7.12 の定量的貢献を可視化） | 運動入力は数エポックで収束するが価値入力は収束に数百エポック要し seed 依存で未収束もある＝入力様式が学習ダイナミクスを質的に変える | 作図のみ（既存ログ） |

**材料チェック結果（2026-09-06、既存 saved.h5 を実測）**：
- F3（PCA）／F4（Fig.6b）／F5（アブレーション）の材料は **P0-2・P0-4 とも充足**。saved.h5 に `state/self|other/hidden`・`self_position`・`other_position`・`q2_hat/prediction` が保存済み。A-2 真 Q はデータ h5 の `other_vision`（3000×101×16×64×3 uint8）＋ A-2 critic で再計算でき、`r4_ve_eval.py` と同じ経路。**新規学習・新規 eval 不要、作図のみ**。
- **不足は F7（予測画像）のみ**：視覚を保存した saved.h5 が存在しない（ハーネスは `--save_targets` に視覚を含めていない）。viz20 データセット（`r2_a1random_a2rl_viz20`・`r3_stay_viz20`）には `self_vision`・`other_vision` の実データがあるので、`other_vision` loader パッチ適用後に viz20 で `test.py --viz` を条件ごとに1回（軽・学習不要）。
- R3-A・R3-stay・v5 の F3 は、それぞれの §P0-3／§P1 評価時に `--save_targets` へ `state` を含めれば同時に取れる（現ハーネスは含めている）。

**この図リストは固定**。生成は P0-3（3シード収束）／P1.75 の結果が出てから、上表のスクリプトを回すだけ。

**コード実測でわかった制約**：
- `saved.h5` の dataset は `create_dataset(path, shp)`（dtype 未指定）＝ **float64**。
- `--test_modes eval` は `data['train']`（≈3000ep）を読む（`gen_data_loader`：`eval_loader = DataLoader(data['train'])`）。
  → `self_vision` の input/truth/prediction 3本で **≈40 GB／条件**。**full split で視覚を save してはいけない。**
- `--save_targets_index` は PNG ダンプ（`_save_vision`）だけをフィルタ。h5（`_save_value`）は全 ep 書く＝サイズ削減にならない。
- `other_vision` は h5 に**ある**（`r2_a1random_a2rl` / `r3_stay` とも uint8, shape 一致）が、`SeqLoader`/`SequenceLoader`/
  runner の modal ループが拾っていない。

**やること**：

1. **`other_vision` を通す加算パッチ（3箇所、`'other_vision' in ...` ガードで後方互換）**：
   - `exp/loader.py` `SeqLoader.__next__`：`if 'other_vision' in self.data: batch['other_vision'] = torch.from_numpy(
     util.scale_vision(util.transpose_vision(self.data['other_vision'][batch_index]))).to(self.device)`
   - `exp/loader.py` `SequenceLoader.__next__`：`x`/`y` に `if 'other_vision' in self.data:` で
     `self.data['other_vision'][:, self.t]` / `[:, self.t+1]`
   - `exp/runner.py` `run_seq` の `collect_data` modal リストに `"other_vision"` を追加
2. **可視化専用の小データセット**（sim 不要・h5 スライス・新規名）：
   ```python
   # data/data/r2_a1random_a2rl_viz20/data.h5 ← r2 の train 先頭 20 ep をコピー
   #   フィールドは self_vision/self_position/self_motion/other_vision/other_position/other_motion
   # 同様に r3_stay_viz20（v5・R3-stay 用）
   ```
   → `self_vision`+`other_vision` で **≈0.4 GB／条件**、seed 0 のみ（図に3シード不要）。
3. **ハーネスに `--viz` を追加**：viz 小データ＋`--eval_seed` 固定＋`--save_vision_img` で `test.py`。
   出力 `data/result/<exp>/0/test/viz20_evalv4v5/save/`（`--test_name` で隔離）と
   `baseline_v4/<label>_predviz/*.jpg`。保存キーと意味：

   | キー | 中身 | 論文図での役割 |
   |---|---|---|
   | `self_vision/truth` | A-1 の実際の次視界 v_{t+1} | 「A-1 実際」 |
   | `self_vision/prediction` | ネットワークの予測 v̂_{t+1}（統合出力。元論文 Fig.4c "Prediction" と同義。h¹ 単独ではない） | 「予測」 |
   | `other_vision/truth` | A-2 の実際の視界 | 「A-2 実際」（横並び比較用） |

4. **「A-2 視点の再構成（"h² が見ているもの"）」は Fig.5 パイプラインで**（§4.5 と同じ作法）：
   `Autoencoder` クラス＋`analyze/analyze_vpt.py`＋`grid` データ。各条件に
   `config/exp/fig5_v4_r3a_direct_1000pretrain.yml` / `fig5_v4_r4ve_exp3parity.yml` / `fig5_v5_base.yml` を追加
   （`fig5_v4_r3stay400.yml` のコピーで `pretrain.exp_config` を差し替えるだけ。各 100ep の Autoencoder、安い）。
   → `other_other` / `other_self` 誤差 ＋ 再構成パネル（`analyze/build_fig5_v4_report.py`）。
   **素朴な per-process decode（`visualize_r3a_predictions.py` Method A/B）は §6 項目8 の既知アーティファクト
   （exp1_l1 でも h² が h¹ をコピーして見える）→ 診断図のみ・留保つき・headline にしない。**

**タイミング**：P0 の学習は止めない。
- **手順1（`exp/loader.py` + `exp/runner.py` パッチ）は P0-2 完了後に適用**する。Python は import 済みモジュールを
  ホットリロードしないので走行中ジョブ（当時 P0-2 / P0-3）に技術的な影響はないが、共有コードを走行中に触る不要リスクを避ける。
  適用後は必ず `python -c "import exp.loader, exp.runner"` で構文チェック。手順1 が必要になるのは手順3 の取得時（評価フェーズ）で、
  P0-2 完了（~9h）より後なので待って問題ない。
  **この同じ P0-2 完了後バッチで `exp/runner.py` に `GeneratorTrainingFeaturePredictionRunner`
  （`MotionGenerationPredictionFeaturePredictionRunner` の同一コードのエイリアス、§P0-2 の注記参照）も追加**し、
  P0-4 / v5-r4 の config をそちらに向ける。既存クラスは不変・追加のみ。
- **手順2（viz データセットの h5 スライス）は今やる**（共有コード・保護データに触れない。新規 dir のみ）。
- 手順3・4 の取得は P0-3 / v5 の評価と同じチェックポイントに対して行う。

---

### P0 成果物チェックリスト

- [ ] `test.py` / `args_util.py` に `--test_name` 分離パッチ（後方互換確認済み）← **最初**
- [ ] P0-0 手順1：`test.py` ＋ 全 analyze スクリプトに forward 直前のシード固定（`--eval_seed`）
- [ ] P0-0 手順2：同一チェックポイント2回評価で `state/self/hidden`・`state/other/hidden` が bit-identical
- [ ] P0-0 手順3：結果 JSON に `eval_seed`・マスク on/off を記録
- [ ] P0-0 手順4：元論文の評価作法（マスク・シード）を調査 → 判断を本ファイルと `v4_experiment_log.md` に記録
- [ ] P0-0 手順5：canonical eval（(A) mask_off か (B) mask_on+N=5）を決定・全条件で統一
- [ ] `analyze/run_eval_v4v5.sh`（ハーネス、二重報告：共通プロトコル＋in-distribution）＋既存 r3_a_direct / r4_ve で再現確認
- [ ] `config/exp/r4_ve_exp3parity.yml` ＋ スモークテスト ＋ 学習 ＋ 評価（非 parity と横並び）＋ 判定文
- [ ] R3-A / R3-stay の seed 1,2 学習（pretrain は seed-0 共通の旨を明記）＋ mean±sd 表（seed 間 sd と評価ノイズを分離）
- [ ] `config/exp/r4_move_exp3parity.yml` 学習 ＋ 収束確認後 1 epoch を選んで評価（非 parity r4_move は対照として保持）
- [ ] 全 run の学習曲線 PNG ＋ 収束の目視コメント
- [ ] `v4_experiment_log.md` に P0-2 の runner 不一致・判定結果、P0-0 の決定性対応を追記
- [ ] P0-6：`other_vision` loader パッチ3箇所（後方互換）＋ `r2_a1random_a2rl_viz20` / `r3_stay_viz20`（h5 スライス）
- [ ] P0-6：ハーネス `--viz`（小データ＋`--eval_seed`＋`--save_vision_img`）＋ 各条件 `fig5_v4_*` / `fig5_v5_base`
- [ ] **視覚は必ず viz 小データセットで保存。full split で `--save_targets self_vision` しない（≈40 GB）**

---

## 4. P1 — v5 ベース段（Visual Encoder も価値の下で学習）

### 4.1 何を答えるか

| | v4 R3-stay（既存） | **v5 ベース段** |
|---|---|---|
| Encoder | exp1_l1_1000 から転移して**凍結** | **ランダム初期化からスクラッチ学習** |
| 問い | 論文の視覚表現を所与として SM は Q から A-1 視覚を予測できるか | dual-encoder + shared-module + 予測損失という**機構そのもの**が、側チャネルが Q のとき自他視覚表現を自己組織化するか |

「元論文の運動入力を価値に置き換えた」と言い切るには、運動レジームで圧力を受けた Encoder を使い続けるのは矛盾がある、というユーザーの指摘への回答。v5 ベース段は元論文の `exp1_mse → exp1_l1` の Q 値版。

**注**：`r3_stay` と `r2_a1random_a2rl` は同サイズ（2058 MB＝同エピソード数）なので、v5 ベース段と R4-move の間に**データ量の交絡はない**。

### 4.2 作る config（2本）— モデル config は既存を流用（新規作成不要）

#### `config/exp/v5_base_mse.yml`
```yaml
model:
    name: SuperpositionNetworkProbeQ
    config: default
    mode: ~
runner: PredictionFeaturePredictionRunner
data:
    name: r3_stay                 # A-2 静止・A-1 ランダム。exp1_mse の self_random_other_stay に対応
load_state_dict_strict: False
p_mask_vision: 0.99
train:
    loss_function: mse_loss
    pretrain: ~                   # ← スクラッチ。v5 の肝
    batch_size: 10
    max_epochs: 200
    save_interval: 10
    test_interval: 10
    optimizer: AdamW
    optim_config:
        default:
            lr: 0.001
            weight_decay: 0.01
        freeze:
            - probe_critic        # A-1 の SAC critic は所与。これだけ凍結
        per_params:
            - name: bias
              args: {weight_decay: 0.0}
            - name: vision_encoder
              args: {weight_decay: 0.001}
            - name: share_lns
              args: {weight_decay: 0.001}
```
（元論文 `exp1_mse.yml` と `per_params`・`weight_decay`・`batch_size`・`max_epochs` を揃える。違いは model=ProbeQ、data=r3_stay、freeze に probe_critic を追加、の3点だけ）

#### `config/exp/v5_base_l1.yml`
```yaml
model:
    name: SuperpositionNetworkProbeQ
    config: default
    mode: ~
runner: PredictionFeaturePredictionRunner
data:
    name: r3_stay
load_state_dict_strict: False
p_mask_vision: 0.99
train:
    loss_function: l1_loss
    pretrain:
        exp_config: v5_base_mse
        seed: 0
        epoch: 200
    batch_size: 10
    max_epochs: 200
    save_interval: 10
    test_interval: 10
    optimizer: AdamW
    optim_config:
        default:
            lr: 0.001
            weight_decay: 0.01
        freeze:                   # 元論文 exp1_l1 と同じ＝デコーダのみ学習
            - superposition_module
            - vision_encoder_module
            - integration_module
            - feature_prediction_module
            - share_lns
            - probe_critic
        per_params:
            - name: bias
              args: {weight_decay: 0.0}
```

### 4.3 実行ゲート（W1 中に）

**壁時計は「実施するか否か」だけを決める。配置（本文 or Appendix）は結果で決めない。**
v5 ベース段は「運動レジームで作った Encoder を使って『価値に置き換えた』と言えるのか」という研究の**前提そのもの**への
直接の回答で、§4.6 の通りどの結果でも問いに答える。**実施したら、結果がどうであれ本文の一結果**（Results の一節）。
Appendix には置かない。

v5 ベース段の run1（`v5_base_mse`、全パラメータ MSE）は v4 で一番重い。**まず 10 epoch だけ試走して壁時計を測る**：
```bash
# max_epochs を 10 にしたコピー config で
python -u train.py --exp_config v5_base_mse_smoke --seed 0 --device cuda:<N>
```
- 全 200 epoch（＋ `v5_base_l1` 200ep）の見積もりが **≲ 7日** → 実施（P1 続行、本文結果）。
- **≳ 10日** → v5 ベース段は**今回は見送り**（Future Work に「compute 時間の制約で未実施のクリーンな検証」と明記）。
  その場合は P0 ＋ P1.5（単流統制）で締め、v4 の結果は「論文の視覚フロントエンドを所与とした場合」と**スコープを明示**して書く。
- 7〜10日はグレー。W1 の GPU 空き状況と他タスクの逼迫で判断（指導教員と相談）。

### 4.4 学習

```bash
python -u train.py --exp_config v5_base_mse --seed 0 --device cuda:<N>   # ~200ep
python -u train.py --exp_config v5_base_l1  --seed 0 --device cuda:<N>   # v5_base_mse ep200 から
```
余裕があれば seed 1,2 も（P1 の判定は seed 0 で先に出してよい）。

### 4.5 評価（v4 R3-stay と横並び）

P0-1 ハーネスを `v5_base_l1` に適用：
- 4軸 R²（`r2_a1random_a2rl` eval split、固定プロトコル）
- 視覚損失（`self_vision` / `feature_prediction_self` / `feature_prediction_other`）を `r3_stay_400`（v4 凍結版）と比較
- Fig5 形式の Encoder-2 視点再構成：`fig5_v4_*.yml` の作法（`Autoencoder` クラス、`analyze/analyze_vpt.py`、`grid` データ）を **`v5_base_l1` を pretrain 元にして**適用。
  → `config/exp/fig5_v5_base.yml`（`fig5_v4_r3stay400.yml` のコピーで `pretrain.exp_config` を `v5_base_l1`、`epoch` を 200 に）
- 学習曲線（P0-5）

### 4.6 判定ルール

| v5 ベース段 vs v4 R3-stay | 読み | 論文での書き方 |
|---|---|---|
| 自他分離・視覚損失が**同等** | フリーズ Encoder は特別な仕事をしていない。機構は価値の下でも transfer する | 強い positive。「Encoder を共学習しても結果は変わらず、機構の拡張性が確認された」 |
| v5 の方が**悪い**（Encoder が分化しない／視覚損失が高いまま） | v4 の見かけの成功は借り物の表現に支えられていた。機構は価値の下で素直に自己組織化しない | 強い（negative）。むしろ面白い境界結果 |
| v5 の方が**良い** | 運動レジームの Encoder は価値には最適でなかった | positive。「価値入力には専用の視覚表現が要る」 |

**どれでも問いに直接答えている。** v5 ベース段は「VE が動くか」に依存しないので、結果が必ず出る＝修論の軸にできる。

### P1 成果物チェックリスト

- [ ] `config/exp/v5_base_mse.yml` / `v5_base_l1.yml` / `v5_base_mse_smoke.yml`
- [ ] 10ep 試走の壁時計計測 ＋ 実施/見送りの判断メモ（配置は結果で決めない：実施＝本文）
- [ ] `v5_base_mse` / `v5_base_l1` 学習ログ ＋ 曲線
- [ ] 4軸 R²・視覚損失・Fig5 形式再構成を v4 R3-stay と並べた表
- [ ] 判定ルールに沿った1段落の結論
- [ ] `v4_experiment_log.md` に v5 節を追加

---

## 4bis. P1.5 — 単流統制（load-bearing。「時間があれば」ではない）

**なぜ必須か**：現状 §1 で h²→other 0.90–0.98 を得ているが、これが「superposition／process-2 の働き」なのか
「A-2 が A-1 視野に映るための視覚エンコード副産物」なのかは**未証明**。(b) の記述（「反応的か、機構由来か」）が
この結果に直結するので、P2 の「時間があれば1つ」枠ではなく P1 と並ぶ必須タスクとする。

**やること**：
1. `model/model.py` に `SuperpositionNetworkProbeQSingleStream` を**新規クラスとして追加**（既存クラスは不変）。
   `SuperpositionNetworkProbeQ` の forward で `os` を使わず `so = integration_module(ss, torch.zeros_like(ss))` に。
   process-2（`ov_enc`・`q2_vec`・SM の other 側呼び出し）を通さない。
2. `model/__init__.py` に import 追加。model config は `SuperpositionNetworkProbeQ/default.yml` を流用（別名 dir にコピー）。
3. `config/exp/r3_a_singlestream.yml`（`r3_a_direct.yml` ベース、`model.name` を差し替え、他は同一）で 200ep 学習。
4. P0-1 ハーネスで `r3_a_direct` と横並び（h¹→self、`self_vision` 損失、`feature_prediction_self`）。

**判定**：
- 単流でも h¹→self・視覚損失が R3-A と**同等** → 「Q レジームでは superposition／process-2 は寄与していない」。
  h²→other の 0.90–0.98 は視覚エンコードの副産物＝**"反応的" と書ける**。境界の強い証拠。
- 単流が**明確に悪い** → process-2 は視覚予測に効いている。h²→other を「機構由来の他者追従」と書ける。

**書き込み先**：`model/model.py`（追加のみ）／`model/__init__.py`（追加のみ）／
`config/model/SuperpositionNetworkProbeQSingleStream/default.yml`（新規）／`config/exp/r3_a_singlestream.yml`（新規）／
`data/result/r3_a_singlestream/0/`（新規）。§2.5 抵触なし。

---

## 4ter. P1.75 — 対称サニティ（A-2 = 赤好き）— **load-bearing に格上げ（2026-09-05）**

### 4ter.0 この実験の位置づけ（2026-09-07 明記・重要）

**P1.75 は原因特定のための対照実験であって、研究の方向転換ではない。** 本研究の問いは
「A-1 が**自分と異なる価値観をもつ他者**を理解できるか」。A-1＝赤好き・A-2＝緑好き（主実験＝P0-4）で
両者の価値観が異なることが問いの前提であり、**A-1＝A-2＝赤好きの P1.75 では問いそのものが成立しない**
（同じ価値観の他者を「理解」しても TT/ST の意味をなさない）。P1.75 の唯一の役割は、
**主実験で VE が機能しなかった原因を「Q 非通約性」と「位置優先」に切り分けること**。

論文での書き方（確定）：
> 主実験（A-1＝赤、A-2＝緑）で VE は A-2 の価値を表現できなかった。原因を特定するため、
> A-2 の報酬を A-1 と一致させ Q 値域を構造的に揃えた対照実験（P1.75）を行った。
> 〔結果に応じて：この条件では VE が収束し A-2 の価値を表現した → 原因は Q 値の通約不可能性／
> この条件でも症状が不変 → 原因は視覚予測目的下での位置優先という、より一般的な境界〕。

以降 P1.75 の数値は主実験の結果を**説明する**ためだけに用い、P1.75 自体を新たな主張の土台にはしない。

**背景**：P0-2/P0-4 の2条件とも VE は A-2 の価値を頑健に表現できなかった（`results_log.md` §4 訂正版）。
原因は少なくとも2つの対立仮説がある：**(a) Q 値が自己と通約不可能**（§7.9 の現行の主張）／
**(b) Q 値域を揃えても、位置を流すほうが視覚予測損失を下げるので位置が優先される**（h²→self が両系列で VE 勾配強度に
単調追従する事実はこちらも支持しうる）。**この2つを区別できるのは対称サニティだけ**。
「Q 非通約性が原因」を結論として書くなら、この実験なしには**検証していない仮説を結論にした**ことになる。
→ **P1.5 と同格の必須タスクに格上げ**。ただし v5 ベース段（研究の核心）を最優先し、**並行して**進める。

**やること（低コスト化：A-1 の actor/critic をそのまま再利用、新規 RL 学習は不要）**：
1. A-2 の報酬を A-1 と同一（赤+1・緑−1）にする。**A-2 の方策には A-1 自身の学習済み `v3_rl_actor.pth`/`v3_rl_critic.pth` をそのまま使う**
   （報酬が同一なので Q 値域は自動的に一致する。新規 SAC 学習が要らない——`collect_data_r2.py` の A-2 側エージェントを
   A-1 用の actor/critic に差し替えるだけ）。
2. 新規データ収集（新規名、例 `r2_a1red_a2red`）。A-1: RandomAgent、A-2: A-1 の actor（赤好き）。3000ep 目安。
3. R3-stay 相当（A-2 静止・この報酬設定）→ R4-move 相当（VE 追加）を回す。exp3-parity runner で（§P0-2 方針を継続）。
   - **base 段は既存 `r3_stay_400/0`（ep400）を再利用**（R3-stay の A-2 は静止＝その報酬設定が base データに無関係）。config は `r4_move_sym.yml`（作成済、`r4_move_exp3parity.yml` から `data.name: r2_a1red_a2red` のみ変更）。`save_interval: 10`（late-5 可）。
4. P0-1 ハーネスで評価：`true_q2 vs zero`、`Q̂²→A-1位置 vs A-2位置`、方向 cos_sim。

**保存設定の事前確認（2026-09-06、Fig.6b 3枚並べ〔元論文 / P0-4 / P1.75〕を確実に作るため）**：
- **A-2 の真 Q は A-1 自身の critic で測る**。P1.75 では A-2 = A-1 の actor なので「A-2 の真の価値」＝ `v3_rl_critic.pth`（A-1 の critic）。`r4_ve_eval.py` は従来 A-2 critic（`v3_rl_a2_critic.pth`）ハードコードだったので、`--a2_critic_path` / `--data_h5` 引数を追加済（既定は従来値＝v4 run は不変）。
  - **P1.75 の eval 起動**：`python analyze/r4_ve_eval.py --exp_config r4_move_sym --epoch <E> --seed 0 --eval_seed 0 --a2_critic_path data/model/v3_rl_critic.pth --data_h5 data/data/r2_a1red_a2red/data.h5`
- **データ h5 に `other_vision` が必要**（A-2 視界 → A-1 critic で真 Q 計算）。`collect_data_r2_symmetric.py` は `other_vision` を `create_dataset`（L151）＋書き込み（L189）で保存する＝確認済。
- **eval の saved.h5 に `q2_hat` が必要**（VE 出力）。ハーネス `run_eval_v4v5.sh` の `--save_targets` は既定で `... q2_hat other_position ...` を含む＝確認済。P1.75 も同ハーネスで evaluate すれば F4 の材料（`q2_hat` ＋ `other_position`）は自動で揃う。
- **F3（PCA）も同時取得**：ハーネスは `--save_targets` に `state` を含むので、P1.75 の saved.h5 に `state/self|other/hidden` が入る＝追加作業不要。

**判定**：
- Q̂² が A-2 の価値（＝ A-1 と同じ意味の「赤への接近度」）を表現するようになる（`true_q2` が zero に対し明確に有利、
  方向が赤を向く）→ **(a) Q 非通約性が原因だった**ことの支持。
- それでも h²/Q̂² が位置で埋まる（値域を揃えても症状不変）→ **(b) 位置優先仮説**の支持。「Q 非通約性」は棄却し、
  「価値駆動エージェントの表現は視覚予測目的の下では原理的に位置に負ける」という、より強い境界主張に書き換える。
- **h²→self の判定は P1.75 が収束していることが前提**（P0-4 は未収束で h²→self がチェックポイント依存＝§7 注記）。
  P1.75 の `check_convergence.py` 判定と h²→self の late-5 を必ずセットで見る。

### 4ter.1 収束/非収束の対比 — 独立した結果（2026-09-07）

**同一構造・同一エポックで、データだけ違う2条件の収束挙動が真逆になった。**

| | P0-4（非対称・主実験） | P1.75（対称・値域一致） |
|---|---|---|
| model / runner / freeze / max_epochs | `SuperpositionNetworkProbeQValueEstimation` / exp3-parity / 同一 / 400 | **すべて同一** |
| base pretrain | `r3_stay_400/0` ep400 | **同一** |
| **違いはデータのみ** | `r2_a1random_a2rl`（A-2＝緑好き。真Q std≈0.009、A-1 の probe-Q スケールと非通約） | `r2_a1red_a2red`（A-2＝A-1 の赤 actor。真Q が A-1 スケールに一致） |
| `feature_prediction_other` 収束（P0-5 基準） | **❌ 未収束**（v_min 8.55@ep191、m_W 11.0、ep348 で 16.7 スパイク、ep400 で 9.14） | **✅ 収束**（v_min 15.08@ep124、m_W 15.57、drift 0.29、trans@3、末尾20ep は 15.3–15.5 で安定） |
| 曲線の形 | 相転移後も振動し続け、擾乱から回復しきらない | 滑らかにプラトーへ |

（絶対値は데이터が違うので比較しない——比較するのは**収束したか否か**と**曲線の形**。）

→ **これまで「収束の遅さ・不安定さ」は実験上の障害として扱ってきたが、P1.75 との対比で初めて『現象』になった：学習の安定性そのものが Q 値域の一致に依存する。** A-2 の価値がネットワークの process-1 が較正されたスケールに載っていれば VE 段は収束し、載っていなければ収束しない。これは (a) Q 非通約性仮説の**独立した傍証**（表現プローブの結果とは別経路）。論文では結果3 の中で「VE の表現が立たない」ことと並べて「そもそも学習が安定しない」ことを1つの結果として書く。

### 4ter.2 評価時の落とし穴（2026-09-07・記録）

**P0-1 late-5 ハーネス（`run_eval_lateckpt.sh` → `run_eval_v4v5.sh`）の [4/5] `r4_ve_eval.py` 呼び出しは
`--a2_critic_path` / `--data_h5` / `--target_pos` を渡していない**＝既定（緑 A-2 critic・`r2_a1random_a2rl`・緑ターゲット）で走る。
P1.75 では**すべて誤り**になる（A-2 は赤好き・critic は A-1・データは `r2_a1red_a2red`）。
- ハーネス由来で **valid なのは**：`test.py` が作る `saved.h5`（`--test_name` に正しいデータセットを渡している）→ **4軸 R²・`q2_position_regression`**。
- ハーネス由来で **invalid なのは**：`_r4ve.json`（`true_q2`/`zero`/`real`/cos_sim/Q̂² 統計）。
- **対処**：P1.75 の `r4_ve_eval` は必ず単体で
  `--a2_critic_path data/model/v3_rl_critic.pth --data_h5 data/data/r2_a1red_a2red/data.h5 --target_pos -9,9` を明示して回す
  （`logs/p1_75_eval_correct.sh`）。ハーネスの `_r4ve.json` は P1.75 では使わない。
- **将来対応**：`run_eval_v4v5.sh` に `--a2_critic_path` 等を通す引数を追加（v4 の既定は不変）。

**タイミング**：v5 ベース段の壁時計判定（W1）と並行してデータ収集を開始し、W2 前半に学習・評価まで進める想定。
v5 ベース段の判定を妨げない。

---

## 5. P2 — 時間があれば（優先度順）

### P2-a. `v5_r4_move` — **P1 相当に格上げ（2026-09-09）。研究の本題に届かせるため必須**

`config/exp/v5_r4_move.yml`（作成済 2026-09-09）：`r4_move_exp3parity.yml` のコピーで **pretrain 元だけ変更**
- `pretrain.exp_config` `r3_stay_400` → `v5_base_l1`、`epoch` `400` → `200`
- 他は全て P0-4 と同一：`runner` exp3-parity、`data` r2_a1random_a2rl、`max_epochs` 400、freeze に `superposition_module` 含む（VE のみ学習）、lr/wd/batch/per_params 同一

**比較対象＝P0-4**（凍結 Encoder・非対称・同一データ・同一 runner・同一エポック）。違いは pretrain 元だけ。

#### 事前登録した判定基準（2026-09-09、結果を見る前に固定）

| 指標 | P0-4（参照） | v5_r4_move の「改善」判定 |
|---|---|---|
| `true_q2 − zero`（late-5 平均±inter-ckpt sd） | +0.00441 ± 0.00048 | **≈0 または負なら改善** |
| `Q̂²→A-2位置` / `→A-1位置` | 0.633 / 0.193 | **差が拡大すれば改善** |
| 方向 cos_sim（緑ターゲット） | −0.80 | **正に転じれば改善** |
| 収束（`fp_other`、P0-5 基準） | ❌ | **✅なら改善** |

#### 事前合意した解釈

- **改善する** → 「凍結 Encoder は VE の失敗に寄与していた」。Encoder と値域の2要因のうち **Encoder も効いていた**ことになる。
- **改善しない** → 「Encoder は VE の失敗の原因ではない」。P1.75 と合わせて **値域の通約不可能性が主因**と確定。

#### standing caveat（結果を見る前に記録）

**v5 の base は h²→other が弱い（common Ridge 0.64 vs 凍結 Encoder 0.86、Fig.4c 場所マップも h² 片軸）。** VE が使える他者情報が少ない状態からのスタート。**改善しなかった場合、「Encoder が原因ではない」のか「base の他者表現が弱すぎて VE が学べない」のかは区別できない。** P0-4／P1.75 の「base h¹→self 0.77」と同種の留保。ただし v5 の h¹ 組織化はむしろ凍結より良い（PC 0.61/0.63 vs R3-stay_400 0.34/0.46）ので、留保は h² 側に限定。

評価は P0-1 ハーネス（late-5）＋ `r4_ve_eval`（`true_q2` 主指標、`--target_pos` 緑、`q2_position_regression`）。saved.h5 は JSON 抽出後すぐ削除（§ディスク運用）。

（対称サニティは §4ter に格上げ・移動済み）

---

## 6. 切ったもの（Future Work セクションに書く）

- **v5 フルチェーン**（v5 ベース → VE → R4-move を通した完全 end-to-end）：compute 時間の制約で未実施。「機構全体が価値の下で自己組織化するか」の完全な検証。
- **R5（自己投影バイアス）**：A-1 の緑嫌悪（`known_confounds.md`）を先に除かないとフロア効果と区別できない。A-1 再学習が前提。
- **A-2 の GAMMA 引き下げ**：値域を揃える対策候補（`v4_experiment_log.md` §7.8、GAMMA≈0.8 が有望との既存見積もり）。
- **多選好の選好分類**：目的地指向 A-2 では preference ≡ position になり position 交絡。行動パターン型選好（CW/CCW/停止）なら交絡なし＝クリーンな拡張実験。
- **`h¹→other` 漏れ（0.585）の原因特定**：限界＋候補説明（Q は状態の関数なので `m_t` ほど h¹ に自己位置保持を強制しない）を1段落で記述して閉じる。

---

## 7. 論文の骨子（結果が出そろった後）

> **注意：貢献1 は P0-3 3シードのうち 2/3 収束（n=2 参考値）で暫定。以下の Thesis statement は
> P1.75（対称サニティ）の数値が出た 2026-09-07 時点で確定した本文。**

**Thesis statement（2026-09-07 確定）**：

*予測学習だけで、A-1 の自己表現は価値入力レジームへ transfer する*——価値入力レジームでも h¹→self（自己位置の線形デコード可能性）は
**0.906**（R3-A_400 収束2シード、n=2 参考値、seed 間差 0.001）で、運動入力（元論文 0.967）よりやや低いが高水準を保つ（貢献1、**2/3 収束**）。
一方、**A-2 静止 base ＋ Q値入力の R3-stay 系列（v4・運動レジーム Encoder を凍結流用）では h¹→self が 0.768 に下がり、3/3 未収束**。
**Encoder も価値入力の下でスクラッチ学習し直すと（v5）この非収束が解消し**（収束・元論文型の滑らかな曲線・fp_other 20 vs 30-60）、
Encoder はむしろ元論文より分化し（cos_sim 0.26 vs 0.48-0.66）、自他分離の**向き**は保つ（状態レベル cross≈0、Fig.5 `other_other<other_self` 成立）。
**→「運動入力で作った Encoder を凍結流用して『価値に置き換えた』と言えるのか」の答えは「言えない」——価値入力には専用の視覚表現が要る。**
ただし **Encoder 再学習は非対称に効く。この非対称は3つの独立した経路で一致して確認された（単一指標ではない）**：

| 観点 | v5 vs v4（凍結 Encoder） | 経路 |
|---|---|---|
| h² の 2D 場所マップ（Fig.4c 相当） | ❌ 片軸（v5 PC 0.87/0.07 vs 凍結 0.81/0.96） | PCA 状態マップ＋PC 総当り、`§5quater (8)` |
| h²→other 位置デコード（128次元 Ridge） | ❌ 低い（0.64 vs 0.86、common プロトコル） | 線形プローブ、`§5quater (3b)` |
| Fig.5 他者視点再現マージン（`other_self − other_other`） | ❌ 1/3.6（0.025 vs 0.091、ep10=ep100 で確認） | Autoencoder VPT、`§5quater (10)(13)` |

**対して自己側は4経路とも改善**：収束（trans@27 vs 0/3）、視覚損失（全指標）、予測画像（0.051 vs 0.077）、h¹ の 2D 場所マップ（PC 0.61/0.63 vs 凍結 0.34/0.46）。

→ **「Encoder 再学習は自己表現の組織化・予測性能・学習安定性を改善するが、他者表現の組織化を一貫して損なう」** という方向性のある非対称。偶然ではなく、独立な測定が同じ方向を指す。
原因は scratch Encoder に確定（R3-stay_400 が同一データ・同一入力・Encoder のみ凍結で 2D マップ〔0.81/0.96〕とマージン〔0.091〕を保つ統制。**報酬配置説・価値入力説・静止データ説はすべて反例で棄却**——R3-A_400〔価値入力でも 2D マップ〕、probe-Q→y 0.55 > →x 0.36〔報酬勾配はむしろ y 優位〕、`results_log.md` §5quater (8)(9)(11)(13)）。
*他者の視覚予測は保たれる*が（h²→other 0.90–0.98、v5 の他者トラッキングは動く他者で未訓練のため cross-dist で 0.65 と低め）、それが process-2／superposition の働きか視覚エンコードの
副産物かは未確定（P1.5 待ち）。

*他者の価値の生成的予測（VE）は、元論文に忠実な条件（P0-4）で明確に失敗する。* 失敗の性質は3つ：
1. **方向が積極的に間違っている**——VE のピーク方向は A-2 の目標と逆を向く（P0-4 緑ターゲット cos_sim **−0.80**、
   対称条件 P1.75 赤ターゲット **−0.46 ± 0.07**、45°以内は両条件とも chance 25% 未満）。値域を揃えても向きは逆のまま。
2. **真 Q を VE 出力に差し替えると視覚予測が悪化する**（P0-4 `true_q2 − zero = +0.00441 ± 0.00048`、late-5 で頑健）。
3. **学習が収束しない**（P0-4、`feature_prediction_other` が基準を外す。base の `r3_stay_400` も 3/3 未収束）。

*原因の切り分け（2要因）*：
- **(i) 対称サニティ P1.75**（A-2 の報酬を A-1 と一致させ Q 値域を構造的に揃えた＝真 Q std 0.009→0.593）：**害（性質2）は Q 値の通約不可能性による。** `true_q2 − zero` は +0.00441 → **+0.00014（ゼロと区別不能）**に戻る。学習も P1.75 ではクリーンに収束する。
- **(ii) P2-a `v5_r4_move`**（Encoder も価値入力でスクラッチ学習、他は P0-4 と同一）：**非収束（性質3）には運動レジーム Encoder も寄与していた。** Encoder を再学習すると `fp_other` が **✅収束**（trans@3、drift 0.066。P0-4 は未収束）。
- → 学習の不安定さは **Q 非通約性 ＋ 運動レジーム Encoder** の両方が原因。

**ただし、Encoder を直しても値域を揃えても、VE の中核的失敗（他者の価値を表現できない）は解消しない（残余の失敗）。**
- P1.75：真 Q は zero を上回らず、方向は依然逆（−0.46）。
- P2-a：v5_r4_move は**収束したのに** `true_q2 − zero` 依然 **+0.00333（頑健）**、方向 **−0.38**、**Q̂² は A-2 でなく A-1 の位置を符号化（逆転：→A-1位置 0.39 ≫ →A-2位置 0.10）**。VE 出力そのものが視覚予測を能動的に悪化（`real − zero = +0.0075`）。
*残余の失敗の主因は「位置優先」*：VE の入力 `ov_enc` は A-1 の視覚から計算されるため、他者の価値を推定するより自己位置を流すほうが視覚予測損失を下げる。h²→self は base 0.08 → VE 段 **0.63**（P2-a の**収束した忠実系列**上で。P0-2 の非忠実性・P0-4 のチェックポイント依存問題がなく安定）。→ **P0-2 限定の「候補」から「支持のある仮説」に格上げ。**
*standing caveat*：v5 base の h²→other は弱い（0.645）。v5_r4_move の VE 段でも 0.660 と改善せず。「Encoder が表現失敗の原因でない」と「base の他者表現が弱すぎて VE が学べない」は完全には区別できない。ただしどちらでも「Encoder を直せば VE が動く」は否定される（v5_r4_move で動いていない）。

*結論として：表現の transfer と他者価値の生成的予測は分離可能であり、後者の失敗は Q 非通約性で境界づけられ、
学習の不安定さには Encoder も寄与するが、それらを解いても「視覚予測目的の下では他者の価値より位置が優先される」という残余の限界が現れる。*

> **P0-4 の未収束と h²→self 不安定（2026-09-06、`results_log.md` §4⑥）**：P0-4 の run は `feature_prediction_other` が
> 収束基準（P0-5）を外す（min 8.55@ep191、ep400 でも 9.14）。最終5チェックポイントで **h²→self が 0.09〜0.53 と振れる**
> （h¹→self/other は不変、h²→other は安定＝不安定なのは h²→self だけ）。
> → **主結果 P0-4 由来の単一チェックポイント値は「定常値」でなく「その付近の値」**として扱い、late-5 平均＋inter-ckpt sd を必ず併記する。
> `true_q2 − zero` はこの窓で頑健（SURVIVES）だが、「VE 圧力で h² が位置で埋まる」の数値的根拠（h²→self の上昇幅）は P0-4 では固定できない
> （**方向は P0-2・P0-4 とも上昇で一致、数値は P0-2〔0.10→0.82・収束済〕のみ確定**）。
> **2026-09-07 追記**：P1.75 は **収束**（trans@3、h²→self 0.056→0.472 ± 0.041 で安定）。h²→self の上昇は P1.75（収束・忠実構造）でも確認できたが、
> Thesis では位置優先を「候補」に留める（VE 出力が位置と相関する他の理由を排除できていないため）。詳細 `results_log.md` §5bis。

| 章 | 内容 |
|---|---|
| 序論 | 他者理解（TT/ST/IT）、Noguchi 機構、その限界（内面の推論は未獲得）、本研究の問い (a)(b)(c) |
| 背景 | superposition network、SAC critic、IRL as ToM、probe-Q ベクトルの動機 |
| 手法 | 環境・A-1/A-2 の選好、probe-Q、モデル各種、**統一評価プロトコル**（P0-0/P0-1、二重報告） |
| 結果1（陽性・機構、P0-3後に確定） | (a) 自己表現の transfer（R3-A **3シード・収束後**）＋ critic_swap（R3′） |
| 結果2（機構の自己組織化 → **研究前提への回答**、P1） | **v5 ベース段（Encoder も Q値入力の下でスクラッチ学習）**：**Encoder を学習し直すと学習の安定性と予測性能が明確に改善**（v5 収束 trans@27・滑らか／R3-stay 3/3 未収束「dip→上昇」病理／fp_other 20 vs 30-60）。Encoder はむしろ元論文より分化（cos_sim 0.26 vs 0.48-0.66）、自他分離は base 水準維持（common h²→self 0.081）。**→「運動入力で作った Encoder を凍結流用して『価値に置き換えた』と言えるのか」＝言えなかった。価値入力には専用の視覚表現が必要だった。** §4.6 は3分岐に無理に当てはめず上記の形で記述（h²→other cross-dist ＋ Fig.5 待ち） |
| 結果3（(b)/(c) の境界、**主結果は P0-4**） | **VE 失敗の3性質**（方向が逆 cos_sim −0.80／真Q が視覚予測を悪化 +0.00441 late-5 頑健／学習非収束）→ **2要因の切り分け**：(i) **対称サニティ P1.75** で「害は Q 非通約性」（値域を揃えると +0.00441→+0.00014・収束）、(ii) **P2-a `v5_r4_move`** で「非収束には運動レジーム Encoder も寄与」（Encoder 再学習で ✅収束）**が、表現の失敗は Encoder を直しても解消しない**（`true_q2−zero` 依然 +0.0033 頑健、方向 −0.38、Q̂² は A-2 でなく **A-1 位置を符号化・逆転**）→ **主因は Q 非通約性 ＋ 位置優先。Encoder は学習安定性にのみ寄与** ／ **位置優先は「支持のある仮説」に格上げ**（P2-a は収束した忠実系列で Q̂²→A-1位置 0.39≫→A-2位置 0.10、h²→self 0.08→0.63、P0-4 のチェックポイント依存問題なし。P0-2 限定でなくなった）／ standing caveat（v5 base h²→other 0.645→VE 段 0.660 と弱いまま。「Encoder が原因でない」と「base 他者表現が弱すぎ」は完全には切り分け不可、ただしどちらでも「Encoder を直せば動く」は否定）／ R2 運動分散崩壊／収束/非収束の対比（§4ter.1）／P1.5 単流統制／Fig5-v4 |
| 考察 | 何が transfer し何がしないか、価値 vs 運動指令、限界（単一環境・手設計報酬・Encoder・h¹→other 漏れ・A-1 緑嫌悪・3シードは SM 部分のみ・R3-A seed0 は「当たり」シード・**P0-4 は未収束で単一値が定常でない・h²→self がチェックポイント依存・P0-4 で頑健に押せるのは `true_q2` のみ・P0-2 は非忠実系列・値域一致下でも他者位置符号化が P0-2 未満の理由は部分的に未解明**）、Future Work |
| 結論 | — |

前回まで「陽性・本命」に据えていた Phase 3 の選好分類は **position 交絡で撤回済み**（`PROJECT_MAP.md` §0/§4.2、`v4_experiment_log.md` に追記予定）。骨子はそれに依存しない。

---

## 8. 共通チェックリスト（各実験で必ず）

- [ ] **着手前に §2.6：書き込み先を列挙 → §2.5 と照合 → ⚠️ があれば回避策と共に報告**
- [ ] 学習起動時に `docs/results_log.md` §1 に1行、評価実行時に §2 に1行（追記専用）
- [ ] `--test_name` パッチ済み（§2.3）。既存 config への `test.py` はパッチ後のみ
- [ ] 評価は `--eval_seed` 固定（P0-0）。同一チェックポイント再評価が bit-identical
- [ ] `--seed` を明示（0/1/2）。主張する数値は 3シード mean ± sd。**seed 間 sd と評価ノイズを分けて報告**
- [ ] 出力 JSON に `gen_result_metadata`（git commit・config 全文・seed・dataset）＋ `eval_seed`・マスク設定
- [ ] 学習後に `plot_training_curve.py` で収束を目視。R4 系は収束領域から 1 epoch を選び1点のみ評価（複数 epoch を並べて選ばない）
- [ ] 新規 exp_config は**新しい名前**。既存 `data/result/<name>/<seed>/` は不変（新規 seed 追加は可・§2.5 注記1）
- [ ] 実行後 `git status`：root 側・削除禁止資産に差分ゼロ。あれば即停止して報告
- [ ] `p_mask_vision` と `runner` の組が意図通りか（P0-2 の教訓）。新規 R4 系は exp3-parity runner
- [ ] freeze リストに `probe_critic` が入っているか（ProbeQ 系）
- [ ] 位置回帰（`regression_baseline_v4.py`）とランドマーク識別（`regression_fixed*.py`）を混同しない

---

## 9. 付録：全タスクの書き込み先一覧（§2.5 と照合済み）

`✅` = §2.5 抵触なし（my_research 配下の新規／意図した追加）。`⚠️` = 保護 dir 内・既存上書きの可能性 → 回避策を併記。
新タスクを足すときはここに追記してから着手する（§2.6）。

| タスク | 書き込み・編集先 | 照合 |
|---|---|---|
| **`--test_name` パッチ**（§2.3、最初） | `test.py`, `args_util.py`（後方互換の追加のみ） | ✅ my_research 配下 |
| **P0-0 手順1**（評価シード固定） | `test.py`, `analyze/{regression_baseline_v4,r4_ve_eval,probe_q_direct_regression,q2_position_regression}.py`, `util.py`（`gen_result_metadata` に項目追加） | ✅ 既存関数への追加 |
| **P0-0 手順2**（決定性チェック） | パッチ後：`data/result/r3_a_direct/0/test/<新test_name>/save/`, `.../test/<新test_name>/log/` | ✅ パッチにより新サブdir。既存 `test/r2_a1random_a2rl/` 不変 |
| **P0-0 手順4**（元論文の作法調査） | 読み取りのみ（root `exp/runner.py` 等）。記録は `docs/*.md` | ✅ |
| **P0-1**（ハーネス） | `analyze/run_eval_v4v5.sh`（新規）／各 config の `test/<新test_name>/`／`data/result/baseline_v4/*.json,png`（追記） | ✅ / 追記可 |
| **P0-2** | `config/exp/r4_ve_exp3parity.yml`（新規）／`data/result/r4_ve_exp3parity/0/`（新規） | ✅ |
| **P0-3** | `data/result/r3_a_direct_1000pretrain/{1,2}/`, `data/result/r3_stay_400/{1,2}/`（**新規 seed**）／pretrain は `exp1_l1_1000/0/model/00200.pth` を**読むだけ** | ✅ §2.5 注記1（seed 分離をコード確認済み・seed 0 不変） |
| **P0-4** | `data/result/r4_move/0/`（非 parity・2026-08-24 完走済・**読み取りのみ**・対照として保持）／`config/exp/r4_move_exp3parity.yml`（新規）／`data/result/r4_move_exp3parity/0/`（新規） | ✅ |
| **P0-5** | `data/result/baseline_v4/*.png`（追記） | ✅ |
| **P1** | `config/exp/{v5_base_mse,v5_base_l1,v5_base_mse_smoke,fig5_v5_base}.yml`（新規）／`config/model/`（fig5 用に既存 yml をコピー）／`data/result/{v5_base_mse,v5_base_l1,v5_base_mse_smoke,fig5_v5_base}/0/`（新規）／`baseline_v4/*.json`（追記） | ✅ |
| **P1.5**（単流統制） | `model/model.py`（**新クラス追加のみ**）／`model/__init__.py`（import 追加）／`config/model/SuperpositionNetworkProbeQSingleStream/default.yml`（新規）／`config/exp/r3_a_singlestream.yml`（新規）／`data/result/r3_a_singlestream/0/`（新規） | ✅ 既存クラス不変 |
| **P2-a** | `config/exp/v5_r4_move.yml`（新規）／`data/result/v5_r4_move/0/`（新規） | ✅ |
| **P2-b**（対称サニティ） | A-2 critic 学習＋データ収集（別途詳細化）／`data/model/*`, `data/data/*`（新規名）／`config/exp/*`（新規）／`data/result/*`（新規） | ✅ 新規名のみ |
| **doc 更新** | `docs/research_plan_v4_v5.md`, `docs/v4_experiment_log.md`, `../../PROJECT_MAP.md`（追記・改訂） | ✅ |

**唯一の ⚠️ だった `test.py` 上書き問題は `--test_name` パッチで解消済み。** パッチを最初に当てることが前提。
