# 公開する物の一覧（投稿前に先生と決めるための材料）

- 作成：2026-10-29（第10弾 §5）。**公開はしていない。一覧を作っただけ**。リポジトリのファイルは何も変えていない。
- 場所はワークツリー `.worktrees/v11/my_research/preference_inference/`（以下 PI/）。ルートは `/home/kusano/superposition`（コンテナの中では `/work`）。
- 大きさは `du -cb`・`stat` で測った（主な値は2回目に自分でも測って一致を確かめた：基準エンコーダ 2,852,733 B，`data/data` 全体 24.4 GB，最終5点のチェックポイント 700 本 2,340,738,320 B，`v11_s1_random` 2.06e9 B，`v12_hide_v11_s2_green` 3.08e9 B，B4 のデータ 1.87e8 B）。
- 「公開に向かないもの」は，他人の名前・メールアドレス，絶対パス（`/home/…`，`/work/…`，`/tmp/…`），鍵・トークン・パスワード，ホスト名，GPU 番号・コンテナ名などの個人の環境の情報，作業の会話（Claude Code）への言及を grep で探した。**鍵・トークン・パスワード・メールアドレス・ホスト名はどの項目にもなかった**。

## 一覧

| 項目 | パス | ファイル数・大きさ | 公開に向かないもの |
|---|---|---|---|
| シミュレータとデータ生成のコード | `simulation/collect_data_v11.py`・`_v11_s2t`・`_v13`・`_v14`・`_paper_a2p`・`_paper_b4`，`make_signrand_paper_c.py`，`make_randompref_v14.py`，読み込む `sim_v10/`（描画のモデル・シェーダーを含む），`v11core/`（`rule_agent`，`pleasure`），`constants.py`。**加えて `analyze/v12/make_hide_a2_v12.py`**（他者遮蔽学習のデータを作る。種 12043） | 62 本，約 105 KB | `sim_v10/__init__.py` 6 行の説明に `/work/…` のパスが1件。乱数の鍵（`v11_s1` など）はコードに書かれている（§7 の鍵）。OpenGL と xvfb が要る |
| 学習のコード | `train_v12.py`（`train_v13ns.py`・`train_v14.py`・`analyze/paper_extra/train_paper_c.py` はこれを流す），`util.py`，`config_util.py`，`args_util.py`，`exp/`（`loader`・`loader_v12`・`loader_v14`・`logger`・`runner`），`model/`（`model.py`・`model_v11`〜`v14`・`modules`・`base`・`util`） | 22 本，約 114 KB | `util.py` 279〜286 行のコメントに作業の会話への言及。結果に `.git/HEAD` のコミットと時刻を書く。`model.py` に論文で使わない別のパッケージを読む行（322・459・552 行，通らない経路） |
| 学習の設定 | `config/model/`（8 本），`config/exp/` の論文で使う設定 121 本（段階I：経験なし・50%・80% 各7，非共有 8，好み切替 5，符号無作為 8，零入力 5；段階II：自由出力 16＋形制限の参考 4＋形制限 16＋他者遮蔽 16，経験80% の自由出力 8，1ビット 14）。基準エンコーダ用にルートの `config/exp/exp1_mse_1000.yml`，PI の `config/exp/exp1_l1_1000.yml`，`sim_v10/config/collect/self_random_other_stay_1000.yml` | 計 154 本，約 249 KB（コードを含む） | **論文で使う段階Iの MSE 期間の設定 8 本すべて**に `pretrain_encoder_from: /work/data/result/exp1_l1_1000/0/model/00200.pth`（同じ行は試しの設定を含め 18 本）。設定はあるが学習していないもの：`v12_s2c_*_s3`，`v14_s2u_pref_{mixed,random}_s3`，`v14_s1_mixedpref_l1_s3` |
| 評価・集計のコード | `analyze/v11`（33 本），`v12`（37），`v13`（18），`v14`（19），`paper_extra`（27 py＋12 sh），`paper`（15 py＋7 md） | 168 本，約 1.05 MB | `/work/…` 6 行（4 本），`/tmp/…` 6 行（2 本，フォントの写し先），「kusano」10 行（「Kusano-san's question」などのコメント，コンテナ名 `kusano_research`），「草野」2 行（`analyze/paper/round3/part_main.md`），GPU 62 行・`CUDA_VISIBLE_DEVICES` 9 行 |
| 起動のスクリプト | PI 直下の `run_*.sh`・`_run_*.sh` | 72 本，約 121 KB | GPU 番号，`/work` のパス，コンテナ名，`/tmp/*.log`。**そのままの公開には向かない**（使い方の説明に置き換える案） |
| 基準エンコーダの重み | ルートの `data/result/exp1_l1_1000/0/model/00200.pth`（L1 期間の ep200。エンコーダは `exp1_mse_1000/0/model/00200.pth`，5,525,201 B と同じ重み） | 1 本，2,852,733 B（md5 549b53f5…） | 中身は重みと optimizer の状態で，パスの文字列はない |
| 学習済みモデル（評価に使った最終5点） | `data/result/` の段階I（経験なし・50%・80%・符号無作為 各組1〜6，非共有・好み切替・零入力 組1〜3）の MSE 期間と L1 期間の ep160〜200，段階II（自由出力・形制限・他者遮蔽・1ビット）の ep160〜200 | **700 本，2.34 GB**（欠けなし）。全チェックポイントなら 9,420 本，37.3 GB | .pth に optimizer の状態が入っている（重みだけにすれば小さくなる）。同じ場所の `train_seed*.log` に `/work/…` のパス |
| 図表を作るスクリプト | `analyze/paper/*.py`（13 本），`analyze/paper_extra/fig_*.py`（3），`s3s5_*.py`（5）；出力 `paper/figs`（27 ファイル，3.4 MB），`paper/tables`（14，53 KB） | — | `analyze/paper/paper_style.py` 5〜6 行に docker のコマンドとコンテナ名，`paper/figs/README.md`・`paper/tables/README.md` 4 行に `/work/.worktrees/…`，`paper/tables/README.md` 19 行に「草野」 |
| 事前登録の記録 | `docs/*.md`（42 本）と `docs` の pdf・html・図（7），`paper/check/*.md`（24 本），git の履歴（ブランチ `feature/v11-self-experience`，main からのコミット 399，docs か paper/check に触れたもの 370，2026-08-12〜10-29） | docs 49 ファイル，9.6 MB | 「草野（さん）」362 行（40 本），「指導教員」30 行，「先生」4 行，Claude（Code）への言及 41 行（25 本），`/home/kusano/…` 18 行（9 本），`/work/…` 12 行，GPU 580 行。**別の利用者のユーザー名が `docs/v6_experiment_log.md` に 6 回**（GPU の使用状況の記録）。git の履歴を出すと作者名とメールアドレス（すべてのコミットが同じ作者）が載る。指示書（`paper_instructions_round4`〜`10`）など 12 ファイルは git に入っていない |

## データ（学習用・評価用のエピソード）

| データ | 大きさ | 作り直してビット単位で一致したかの記録 |
|---|---|---|
| `v11_s1_random`，`v11_s1_mixed` | 各 2.06 GB | **ある**：先頭 3 エピソードを2回集めて一致，本収集の先頭 3 エピソードとも一致（`docs/v11_experiment_log.md` §7.1，`v11_results_log.md`） |
| `v13_s1_mixed80` | 2.06 GB | **ある**（同じく先頭 3 エピソード，`v13_experiment_log.md` §1） |
| `v11_s2t_green`・`_swap`・`_red` | 各 179 MB | **ある**（2回集めた先頭 3 エピソードが一致，`v11_experiment_log.md` §7.37） |
| `v11_s2_green`，`v11_s2_pref` | 各 2.06 GB | **ある**（先頭 3 エピソード，`v11_experiment_log.md` §7.40；評価用の先頭 5 エピソードが `paper_extra_log.md` §29 でも一致） |
| `v14_s1_mixedpref` | 2.06 GB | **ある**（`v14_experiment_log.md` §17：赤好きのエピソードが経験50% のデータと全配列で一致） |
| `v14_s1_randompref` | 2.06 GB | `v11_s1_random` の写しで，全配列の一致を確認（§24）。論文では使っていない |
| 符号無作為の符号の表 `signs.h5` | 28 KB | 作り直しの記録はない。符号以外の配列の一致（`paper_extra_log.md` §1.4）と md5 の記録（§34）があり，現物の md5 と一致 |
| A-2 0.8 の評価データ（`paper_s2t_green_a2p80`，`paper_s2_pref_a2p80`） | 各 179 MB | 手順の確認だけ（確率 0.5 で流すと元のデータと 300/300 一致，§7.3・§8）。0.8 の版を2回作った記録はない |
| 追跡のできる評価データ（`paper_s2t_green_a1p80`，`paper_s2_pref_a1p80`） | 各 179 MB | 手順の確認だけ（A-1 をランダムに戻すと元の先頭 5 エピソードと一致，§29）。0.8 の版を2回作った記録はない |
| 他者遮蔽学習のデータ（`v12_hide_v11_s2_*`，`v12_hide_v11_s2t_*`） | 2.9 GB × 2，268 MB × 3 | 2回作った記録はない（`v12_experiment_log.md` §45 は描き直した視界と保存値の一致 37500/37500 だけ） |
| 零入力条件 | ― | ファイルはない（読み込みのときに `v13_s1_mixed80` の快・不快を0にする） |
| 基準エンコーダの学習データ（ルートの `data/data/self_random_other_stay_1000`） | 1.37 GB | 記録はない |

- 合計：`data/data` 全体 24.4 GB（他者遮蔽学習のデータを除くと 15.7 GB）。1 GB を超えるのは 2.06 GB のデータ 7 個，他者遮蔽学習の 2.9 GB 2 個，基準エンコーダのデータ 1.37 GB。
- 学習の作り直し：記録があるのは ep0 の重みの一致（CPU）と，`v13_experiment_log.md` §13 の段階II の1本の 10 エポック（GPU でビット単位で一致）だけ。全部の学習を作り直した記録はない。

## 決める必要があること（材料）

1. データは配るか，生成のコードと鍵だけにするか（作り直しの記録がないデータ：A-2 0.8，追跡のできる評価データ，他者遮蔽学習，基準エンコーダの学習データ）。
2. チェックポイントは最終5点だけ（2.34 GB，optimizer を除けばさらに小さい）か，全部（37.3 GB）か。
3. 作業記録（docs）をそのまま出すか（人名・別の利用者のユーザー名・絶対パス・作業の会話への言及を含む），事前登録の節だけを抜き出すか。git の履歴を出すと作者のメールアドレスが載る。
4. コードと設定の絶対パス（`/work/…`）と起動のスクリプト（GPU 番号・コンテナ名）の扱い。
