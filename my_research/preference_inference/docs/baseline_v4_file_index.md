# `data/result/baseline_v4/` file index

Every analysis/result file this project has written lands in this one flat
directory, and filenames alone don't always disambiguate which run/config/
bugfix state produced them. This is a manual index, current as of
2026-08-15. New scripts should prefer `util.gen_result_metadata()` (git
commit + config snapshot + seed + dataset, merged into the output JSON) so
this kind of external index becomes less necessary going forward — but
existing files below predate that and are not being retroactively
regenerated.

**If a JSON file listed below has no `config_content`/`git_commit` field,**
it predates `util.gen_result_metadata()` and its provenance is documented
here only, not self-contained in the file.

## R3-A: the file that matters most right now

| File | What it is |
|---|---|
| `r3_a_direct_r2.json` / `.txt` | **ORIGINAL** R3-A (RL) 4-axis R², from the checkpoint trained with the pre-bugfix `load_pretrain()` (weight_ih either fully shape-matched or fully skipped — no partial-column transfer existed yet; effectively full-random-init for the SM's new probe-Q input columns). h1→self=0.9095, h1→other=0.5820. **This is the number the user explicitly declared still valid** despite the later-discovered 66-column bug, because that bug was introduced only afterward and never touched this checkpoint. |
| `r3_a_direct_v2_r2.json` / `.txt` | **CORRECTED** R3-A (RL) 4-axis R², from the checkpoint retrained after fixing the 66-column partial-weight-transfer bug in `util.load_pretrain()` (`VISION_ENC_DIM=64` fix). h1→self=0.9086, h1→other=0.5849 — essentially identical to the original, confirming the bug fix didn't materially change results. Has full `gen_result_metadata()` provenance (git commit, config snapshot, seed=0, dataset=r2_a1random_a2rl). |
| `r3_a_direct_cycler_v2_r2.json` / `.txt` | R3-A-**cycler-control** 4-axis R² (control experiment: A-2 = deterministic Cycler agent instead of the trained RL policy), same corrected weight-transfer code as `_v2` above (cycler control only ever ran under the fixed code — there is no pre-fix "v1" for it). h1→self=0.8219, h1→other=**0.9062** (higher than the RL version, not lower) — this result is what falsified the "elevated h1→other is caused by A-2's RL policy" hypothesis. Has full provenance metadata. |
| `r3_a_direct_prediction_viz_ep0/1/2.png` | `visualize_r3a_predictions.py` run against `r3_a_direct` (the **original**, pre-bugfix checkpoint) using Method A (FPM→decoder) and Method B (integration(h,zero)→decoder). Superseded finding: both methods were later shown to be invalid decode paths (see `exp1_l1_sanity_prediction_viz_ep*.png` below) — these images should not be used to judge R3-A's self/other separation. |

## exp1_l1 sanity checks (methodology validation, not R3-A results)

| File | What it is |
|---|---|
| `exp1_l1_r2.json` / `.txt` | Baseline 4-axis R² for the root-repo's own `exp1_l1` checkpoint (§5.0.2 baseline re-measurement), used as the "known-good" reference point throughout. |
| `exp1_l1_sanity_prediction_viz_ep0/1/2.png` | `visualize_r3a_predictions.py` run on `exp1_l1` (not R3-A) as a methodology sanity check. Showed the SAME "h2 copies h1" artifact that R3-A showed under Method A/B — proved the *visualization method itself* (not R3-A) was invalid, since exp1_l1 is known-good (h2→other=0.9725) but still showed the artifact. |

## Fig.5 (viewpoint-taking) — all three attempts, now deprecated

**Do not use any of these to judge the paper's Fig.5 claim.** The claim is
already verified via the original repo's own pipeline: see
`docs/known_confounds.md` § "Fig.5 (viewpoint-taking) reproduction:
corrected after root-repo audit" and
`data/result/exp2/0/test/grid/save/vpt/1/histogram/result.txt` (root repo,
not under `my_research/`). The files below are kept only as a record of a
failed independent-reproduction attempt (`analyze/train_fig5_decoder.py`
and `analyze/eval_fig5_decoder_quant.py`, both header-flagged deprecated).

| File | What it is |
|---|---|
| `exp1_l1_fig5_decoder.pth` / `_quant.json` / `_viz_ep0/1/2.png` | **v1**: first attempt. Trained on `r2_a1random_a2rl` (wrong data — A-2 moving via RL, not the paper's own training distribution) for only 15 epochs×625 batches. Failed to reproduce the paper's direction. |
| `exp1_l1_v2_fig5_decoder.pth` / `_quant.json` / `_viz_ep0/1/2.png` | **v2**: switched training data to `self_random_other_stay` (correct, paper-matching) and scale (~100k steps), but had an undiscovered bug: `self_random_other_stay` is stored as float32 already in `[0,1]`, and the training loop unconditionally divided by 255 again, crushing all training targets to a near-constant value near -1. Decoder collapsed to outputting a near-constant image (self_self eval error 1.068 despite reported train loss 0.0010 — the giveaway). |
| `exp1_l1_v3_fig5_decoder.pth` / `_quant.json` / `_viz_ep0/1/2.png` | **v3**: bug fixed (removed the erroneous manual `/255`, let `util.scale_vision`'s own dtype check handle both uint8 and float32 sources). self_self error (0.0713) now matches train loss (0.0715), confirming the pipeline is internally consistent — but the paper's claim still did not reproduce (other_other=0.5506 > other_self=0.3536). Root-repo audit later explained why: wrong model class (`VisionDecoderModule` autoencoder bolt-on vs. the paper's actual `Autoencoder` class), no weight_decay regularization (paper uses 3.0), and wrong eval dataset (natural trajectory vs. the paper's exhaustive `grid`). |

## R2 gate investigation (motion generator / A-2 RL policy)

| File | What it is |
|---|---|
| `r2_a2_rl_r2.json` / `.txt` | R2's 4-axis R² with A-2 = trained RL Green-policy (vs. exp3's Cycler A-2). Failing result that triggered the root-cause investigation: h2→self=0.3421 (need <0.15). |
| `r2_full_report.json` | Consolidated R2 investigation report (position correlation check + motion variance check + MG accuracy, combined). |
| `r2_mg_accuracy.json` | Motion Generator (MG) prediction accuracy under R2's conditions. |
| `r2_mg_error_by_position.json` | MG error broken down spatially — used to rule out "regional/localized MG failure" as the cause (error was uniform, not concentrated). |
| `r2_motion_variance_check.json` | A-2's `other_motion` std_y under RL policy (0.0154) vs. A-1's own reference (0.678) and exp3's Cycler A-2 (0.690) — the ~44x collapse that is R2's confirmed root cause ("A value-driven policy structurally starves the Motion Generator", see `known_confounds.md`). |
| `r2_position_correlation_check.json` | A-1/A-2 position correlation under R2 (found near-zero, <0.003 R² ceiling) — ruled out as an alternative explanation before landing on the motion-variance-collapse root cause. |
| `r2_vs_exp3_motion_histograms.png` | Visual comparison: A-2 `other_motion` distribution under R2 (RL policy) vs. exp3 (Cycler) — shows the variance collapse directly. |
| `r2_vs_exp3_position_overlay.png` | Visual comparison: A-1/A-2 position coverage, R2 vs. exp3. |

## h1→other root-cause investigation (R3-A elevated cross-decode)

| File | What it is |
|---|---|
| `a1_visibility_in_a2_fov.json` | Checked whether A-1 is directly visible in A-2's field of view often enough to explain h1→other=0.58 via a simple visibility confound. Inconclusive/imprecise per the conversation record — not the final answer (superseded by the Cycler-control result, which pointed to an architectural cause instead). |
| `r3a_temporal_dependence_check.json` | Checked for a time-lagged correlation between A-1 and A-2 behavior as an alternative explanation. Refuted. |

## Probe-Q design decisions (pre-R3)

| File | What it is |
|---|---|
| `a1_q_vision_correlation.json` / `_png` (`a1_q_vs_vision_correlation.png`) | Pre-R3 check confirming A-1's probe-Q vector actually correlates with its own vision input (Q-vision correspondence), not flat/degenerate. |
| `a2_q_variance_real_data.json` / `.png` | Confirmed A-2's Q is not flat on real (non-synthetic) trajectory data — a precondition for the probe-Q design to carry any signal at all. |
| `q_normalization_simulation.json` | Simulated tanh vs. linear-clip normalization of the probe-Q vector; basis for the "tanh, not linear clip" decision recorded in `known_confounds.md` (tanh retains ~5.5% of A-2's variance vs. ~35% for linear clip, but A-1 — the only side R3 actually uses — keeps ~2x more variance under tanh: 0.629 vs 0.333). |
| `a1_q_spatial_map.png`, `a2_q_spatial_map.png` | Spatial maps (probe-Q value over a position grid) for the canonical A-1/A-2 seeds. |
| `a1_heatmap.png`, `a1_heatmap_stochastic.png`, `a2_heatmap.png`, `a2_heatmap_stochastic.png` | Position-coverage heatmaps for A-1/A-2, deterministic vs. stochastic policy rollout. |
| `a1_seed0_heatmap_stochastic.png`, `a1_seed0_q_spatial_map.png`, `a1_seed1_heatmap_stochastic.png`, `a1_seed1_q_spatial_map.png` | A-1 3-seed retrain comparison (continuous-reward redesign). Seed0 adopted as canonical; seed1 kept as a comparison point (both healthy, but non-overlapping Q ranges — see `known_confounds.md`). |
| `a2_seed0_heatmap_stochastic.png`, `a2_seed0_q_spatial_map.png`, `a2_seed1_q_spatial_map.png`, `a2_seed2_q_spatial_map.png` | A-2 3-seed retrain comparison. Only seed0 produced a healthy (non-collapsed) critic; seed1/seed2 collapsed. Seed0 adopted as canonical A-2. |

## v3-era baseline (pre-R2/R3, kept for continuity)

| File | What it is |
|---|---|
| `exp3_r2.json` / `.txt` | Baseline 4-axis R² for the root-repo's own `exp3` checkpoint (Cycler A-2, motion-generator architecture) — the other "known-good" reference point alongside `exp1_l1_r2.json`. |
| `v3_b_mgve_r2.json` / `.txt` | 4-axis R² for the v3-era "Approach B" MGVE (motion-generator + value-estimator) experiment, predating the R2/R3 gated ladder. |
