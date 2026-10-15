# v10 数値の一次記録

追記専用（行を消さない）。各行に：日付、段、実験ID（config・シード・エポック）、git commit、学習・評価のシード、データセットと split、配置（元／入れ替え／混在）、値（late-5 平均 ± チェックポイント間 sd）、収束判定（`min(B)` 併記）、出力ファイル、図の目視確認の有無。解釈は `v10_experiment_log.md` に書く。

## 1. 環境・資産の確認

| 日付 | 項目 | 値 | commit | 備考 |
|---|---|---|---|---|
| 2026-10-15 | ディスク空き（/home） | 146 GB / 894 GB（84% 使用） | 11477f0 | `df -h` |
| 2026-10-15 | GPU | GPU0 他ユーザー（1.3GB）、GPU1〜7 空き | 11477f0 | `nvidia-smi` |
| 2026-10-15 | 鏡映対称 \|緑(x,y)−赤(x,−y)\| 最大（7×7 格子、元の配置、他者退避） | 0.0 | 11477f0 | 読み取り専用の描画テスト |
| 2026-10-15 | 4隅での色の割合（A-1 を (−9,9)/(−9,−9)/(9,−9)/(9,9) に置く） | 赤 0.0938 / 緑 0.0938 / 青 0.0938 / シアン 0.0938（各地点で最大の色） | 11477f0 | 同上 |

## 2. 関門 6.0（2026-10-15）

実行時の HEAD は `f63b69b`（v10 のコードは未コミットの作業ツリー。次のコミット `6.0 完了` に含む）。シミュレータ `sim_v10/`、GPU1、コンテナ内 xvfb。

| 日付 | 確認 | 値 | 判定 | 出力 | 目視 |
|---|---|---|---|---|---|
| 2026-10-15 | 1. `original` で `r2_a1random_a2rl` の train 先頭 3 ep を同じシード（md5 `r2_a1random_a2rl` → 2205419040）・同じ A-2 actor（`v3_rl_a2_actor.pth`）で再収集 | `self_vision`・`other_vision`・`self_motion`・`self_position`・`other_motion`・`other_position` の6配列すべて `array_equal=True`、max\|diff\|=0 | **合格** | `data/result/v10_swap/gate60/bitcheck_original.h5`（git 管理外） | — |
| 2026-10-15 | 2. 21×21 格子（A-2 は (50,50)）で \|赤(swap)−緑(orig)\|、\|緑(swap)−赤(orig)\|、\|青(swap)−青(orig)\|、\|シアン(swap)−シアン(orig)\| の最大 | 0.0 / 0.0 / 0.0 / 0.0（画素数で 0） | **合格** | `gate60/gate60_layout_check.json`、`gate60_fractions.npz` | — |
| 2026-10-15 | 配置を `swap_rg` にして `original` に戻したときの視界（441 点） | 全点で bit-identical | 合格（追加の確認） | 同上 | — |
| 2026-10-15 | 3. 報酬マップ（4色の割合・A-1 の報酬、両配置） | A-1 の報酬の範囲は両配置とも [−0.164, +0.164]。入れ替えで赤・緑の割合と報酬が上下に反転、青・シアンは不変 | 図を作成 | `gate60/gate60_fraction_reward_maps.png` | **目視確認済み** |
| 2026-10-15 | 4. swap の (x,y) の視界と original の (x,−y) の視界（§4.2 の確認） | 441 点中、一致する点 **0**。異なる画素数 最小 18・平均 359.4。赤・緑の割合の差は最大 0.0029（3 画素、格子の端の点での描画の非対称） | **一致しない（確認どおり）** | `gate60/gate60_view_comparison.png`、`gate60_view_examples.npz` | **目視確認済み** |

### 2.1 P2：`test_v10.py` の出力先の確認（2026-10-15、HEAD `f63b69b`）

- `analyze/v10/test_v10.py --exp_config r3_a_direct_1000pretrain_400 --seed 0 --test_epoch 400 --test_data_name r2_a1random_a2rl_viz20 --test_modes eval --eval_seed 0 --cudnn_deterministic --save_targets state self_vision other_vision self_position other_position`、GPU1、`--out_dir data/result/v10_swap/p2_check/run_{a,b}` で2回。
- 実行前後で `data/result`（`v10_swap` を除く）・`data/model`・`data/data` の全ファイル・ディレクトリのパス・サイズ・更新時刻（7738 行）を比較 → **差分なし**（保護対象の中に新しいファイル・ディレクトリは作られていない）。
- 2 回の `saved.h5` の全 13 データセットが **bit-identical**。`self_vision/{input,truth,prediction}`、`other_vision/{input,truth}`、`state/{self,other}/{hidden,cell}`、位置が保存されることを確認。
- 確認後、`p2_check/` の `saved.h5`（2 本、計 243MB）は削除した（このファイルから作る図・JSON は無い）。

## 3. 段階A

（未実施）
