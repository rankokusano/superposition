"""
S4 prediction-image comparison (2026-10-01, user request): does the
SS28.2 distance-binned numeric finding (true_r helps increasingly at
range, constant does not share the trend) show up as a visible difference
in actual decoded images, not just aggregate losses? "Do not judge by
numbers alone" -- same standard applied to S2 (plot_pred_images_v6.py).

Replays a handful of episodes containing a >=15-distance-to-Green frame
(the SS28.2 far bin) from t=0 up through the target timestep, under all 4
oracle modes (true_r/wrong_r/zero/constant), and renders self_vision
truth/prediction for each mode side by side, plus other_vision truth for
reference. There is no "real"/VE mode in v6 S4 (no VE is trained -- see
docs/v6_experiment_log.md Sec.33), unlike v4's r4_ve_eval.py.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/plot_oracle_pred_images_v6.py \
        --exp_config v6_s3_base_l1 --epoch 200 --label v6_s3_base_l1_oracle
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if '/work' not in sys.path:
    sys.path.insert(0, '/work')
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import model as models  # noqa
from model import util as model_util  # noqa
import util  # noqa
from my_research.preference_inference.model.rl_agent_sac_v6 import A1_TRUE_R, A2_TRUE_R  # noqa
from oracle_eval_v6 import compute_q2, MODES, DATA_H5, Args, GREEN_POS  # noqa

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'data/result/v6_baseline'
# (episode, target_t) pairs picked from the >=15-distance-to-Green bin
# (docs/v6_experiment_log.md Sec.35.3) -- all such frames fall at t<=8 in
# this dataset (A-2 starts far from Green and reaches it within ~10
# steps), found via a one-off scan of other_position vs GREEN_POS.
FAR_FRAMES = [(0, 5), (4, 5), (16, 5), (40, 5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', default='v6_s3_base_l1')
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label', default='v6_s3_base_l1_oracle')
    ap.add_argument('--data_h5', default=DATA_H5)
    args = ap.parse_args()

    exp_config = util.gen_exp_config(Args(args.exp_config, args.seed))
    model_config = util.gen_model_config(exp_config)
    _, model_dir, _ = util.gen_dirs(Args(args.exp_config, args.seed), test=False)
    model = getattr(models, exp_config.model.name)(model_config)
    model.to(DEVICE)
    util.load_model(model_dir, args.epoch, model)
    model.eval()

    p_mask_vision = exp_config.p_mask_vision
    K = model.probe_actions.size(0)
    eps = sorted(set(e for e, t in FAR_FRAMES))
    max_t = max(t for e, t in FAR_FRAMES)

    with h5py.File(args.data_h5, 'r') as f:
        sv_all = f['train/self_vision'][eps, :max_t + 1]
        ov_all = f['train/other_vision'][eps, :max_t + 1]
        op_all = f['train/other_position'][eps, :max_t + 1]

    ep_to_row = {e: i for i, e in enumerate(eps)}
    bsz = len(eps)
    captured = {mode: {} for mode in MODES}  # mode -> {episode: image array (H,W,3)}
    captured_truth = {}
    captured_other_truth = {}

    with torch.no_grad():
        model.init_state(bsz)
        init_state_snapshot = {k: v for k, v in model.superposition_module.state.items()}
        per_mode_state = {mode: init_state_snapshot for mode in MODES}

        for t in range(max_t + 1):
            sv_np = sv_all[:, t].astype(np.float32)
            if sv_np.max() > 1.5:
                sv_np = sv_np / 255.0
            sv_t = torch.tensor(util.scale_vision(sv_np)).permute(0, 3, 1, 2).float().to(DEVICE)
            sv_raw = (sv_t + 1) / 2

            ov_np = ov_all[:, t].astype(np.float32)
            if ov_np.max() > 1.5:
                ov_np = ov_np / 255.0
            ov_t = torch.tensor(util.scale_vision(ov_np)).permute(0, 3, 1, 2).float().to(DEVICE)
            ov_raw = (ov_t + 1) / 2

            p_mask = 0.0 if t == 0 else p_mask_vision

            sv_enc = model.self_vision_encoder_module(sv_t)
            ov_enc = model.other_vision_encoder_module(ov_t)  # primary design: real other_vision
            q1_vec = model.compute_probe_q(sv_raw)
            sv_enc_m = model_util.mask(sv_enc, p_mask)
            ov_enc_m = model_util.mask(ov_enc, p_mask)

            q2_true_r = compute_q2(model, ov_raw, 'true_r', A2_TRUE_R, bsz, K)
            q2_by_mode = {
                'true_r': q2_true_r,
                'wrong_r': compute_q2(model, ov_raw, 'wrong_r', A1_TRUE_R, bsz, K),
                'zero': compute_q2(model, ov_raw, 'zero', None, bsz, K),
                'constant': q2_true_r.mean(dim=1, keepdim=True).expand(-1, K),
            }

            preds = {}
            for mode in MODES:
                model.superposition_module.state = per_mode_state[mode]
                ss, os_ = model.superposition_module(sv_enc_m, q1_vec, ov_enc_m, q2_by_mode[mode])
                per_mode_state[mode] = model.superposition_module.state
                so = model.integration_module(ss, os_)
                preds[mode] = model.vision_decoder_module(so)

            for e, target_t in FAR_FRAMES:
                if t != target_t:
                    continue
                row = ep_to_row[e]
                captured_truth[e] = sv_t[row].permute(1, 2, 0).cpu().numpy()
                captured_other_truth[e] = ov_t[row].permute(1, 2, 0).cpu().numpy()
                for mode in MODES:
                    captured[mode][e] = preds[mode][row].permute(1, 2, 0).cpu().numpy()

    def show(ax, img):
        img = np.clip((img + 1) / 2, 0, 1) if img.min() < 0 else np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])

    row_labels = ['self truth', 'other truth'] + [f'self pred ({m})' for m in MODES]
    n_rows = len(row_labels)
    n_cols = len(FAR_FRAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.2 * n_rows))
    for j, (e, t) in enumerate(FAR_FRAMES):
        d = float(np.linalg.norm(GREEN_POS - op_all[ep_to_row[e], t]))
        axes[0, j].set_title(f'ep{e} t={t}\ndist-to-Green={d:.1f}', fontsize=9)
        show(axes[0, j], captured_truth[e])
        show(axes[1, j], captured_other_truth[e])
        for i, mode in enumerate(MODES):
            show(axes[2 + i, j], captured[mode][e])
    for i, lbl in enumerate(row_labels):
        axes[i, 0].set_ylabel(lbl, fontsize=8)

    plt.suptitle(f'{args.label} ep{args.epoch}: S4 far-distance (>=15) frame predictions by mode',
                 fontsize=12)
    plt.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{args.label}_far_pred_images_ep{args.epoch}.png')
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f'Saved: {out_path}')

    # --- Fig 2: |true_r_pred - zero_pred| diff heatmap -- the per-mode
    # predictions above look visually near-identical (the SS35 effect size
    # is a few % of the overall loss), so this makes the (real, per SS28.2's
    # inter-checkpoint-stable numbers) difference visible at all. ---
    fig2, axes2 = plt.subplots(2, len(FAR_FRAMES), figsize=(2.4 * len(FAR_FRAMES), 4.4))
    for j, (e, t) in enumerate(FAR_FRAMES):
        diff = np.abs(captured['true_r'][e] - captured['zero'][e]).mean(axis=-1)
        show(axes2[0, j], captured_truth[e])
        axes2[0, j].set_title(f'ep{e} t={t}', fontsize=9)
        im = axes2[1, j].imshow(diff, cmap='inferno')
        axes2[1, j].set_xticks([]); axes2[1, j].set_yticks([])
        plt.colorbar(im, ax=axes2[1, j], fraction=0.046)
    axes2[0, 0].set_ylabel('self truth', fontsize=8)
    axes2[1, 0].set_ylabel('|true_r - zero| pred diff', fontsize=8)
    plt.suptitle(f'{args.label} ep{args.epoch}: pixel-level true_r vs zero difference (far frames)',
                 fontsize=11)
    plt.tight_layout()
    out_path2 = os.path.join(SAVE_DIR, f'{args.label}_far_pred_diff_ep{args.epoch}.png')
    plt.savefig(out_path2, dpi=130)
    plt.close()
    print(f'Saved: {out_path2}')


if __name__ == '__main__':
    main()
