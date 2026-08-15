"""
v4 R3-A visualization (2026-08-14, revised after methodology concern).

Decodes h1 and h2 into images via TWO different paths, both reported so
the decode method itself can be validated (not assumed correct) before
trusting what it shows about R3-A:

  METHOD A (original, now suspect): vision_decoder_module(FPM(h)) for
    h=ss or h=os directly. FPM's output was only ever trained against a
    feature-prediction MSE loss (predict sv_enc from ss, ov_enc from os)
    -- it was NEVER trained to be fed into vision_decoder_module. That
    module was only ever trained on so=integration_module(ss,os)'s
    output distribution. Feeding it FPM's output is architecturally
    valid (dimensions match, 64-dim) but may be badly out-of-distribution
    -- i.e. what looked like "h2 copies h1" could be a decoder artifact,
    not a real property of h2.

  METHOD B (integration-routed, more likely faithful): so computed by
    integration_module(ss, zeros) for "self-only" and
    integration_module(zeros, os) for "other-only", THEN through
    vision_decoder_module. This stays on the actual trained self_vision
    reconstruction path (integration -> decoder), which IS the path the
    reconstruction loss optimized end-to-end, unlike Method A. Still not
    a perfect match to training (training's dropout zeroes individual
    dimensions of ss/os independently at p=0.5, not the whole vector at
    once), but far closer to in-distribution than Method A.

Run on exp1_l1 (h2->other=0.9725, KNOWN good self/other separation, no
extra training needed -- checkpoint already exists) as a methodology
sanity check: if Method A/B correctly show a self-like h1 reconstruction
and an other-like h2 reconstruction for exp1_l1, the method is trustworthy
and R3-A's copy artifact is real. If either method ALSO shows copying for
exp1_l1, that method is not diagnostic and the corresponding R3-A finding
should be discounted.

Works with any exp_config/model class: self_motion is loaded and passed
into the input dict, ignored by models that don't read it (R3-A) and
required by ones that do (exp1_l1 / SuperpositionNetworkFeaturePrediction).

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/visualize_r3a_predictions.py \
        --exp_config exp1_l1 --epoch 200 --n_episodes 3 --label exp1_l1_sanity
    python analyze/visualize_r3a_predictions.py \
        --exp_config r3_a_direct --epoch 200 --n_episodes 3
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
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import model as models  # noqa
import util  # noqa

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
SAVE_DIR = 'data/result/baseline_v4'
TIMESTEPS = [10, 30, 50, 70, 90]
SEED = 12345


class Args:
    def __init__(self, exp_config, seed):
        self.exp_config = exp_config
        self.seed = seed


def to_img(v_scaled_m1_1):
    v = ((v_scaled_m1_1 + 1) / 2).clamp(0, 1)
    v = v.permute(1, 2, 0).cpu().numpy()
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', default='r3_a_direct')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--n_episodes', type=int, default=3)
    parser.add_argument('--label', default=None)
    args = parser.parse_args()
    label = args.label or args.exp_config

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    exp_config = util.gen_exp_config(Args(args.exp_config, 0))
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(Args(args.exp_config, 0), test=False)

    model = getattr(models, exp_config.model.name)(model_config)
    model.to(DEVICE)
    util.load_model(model_dir, args.epoch, model)
    model.eval()

    p_mask_vision = exp_config.p_mask_vision if 'p_mask_vision' in exp_config else 0.99

    with h5py.File(DATA_H5, 'r') as f:
        n_data, seq_len = f['train/self_vision'].shape[:2]
        ep_indices = np.random.RandomState(SEED).choice(n_data, size=args.n_episodes, replace=False)
        sv_all = np.stack([f['train/self_vision'][int(i)] for i in ep_indices])
        ov_all = np.stack([f['train/other_vision'][int(i)] for i in ep_indices])
        sm_all = np.stack([f['train/self_motion'][int(i)] for i in ep_indices])

    os.makedirs(SAVE_DIR, exist_ok=True)

    for e in range(args.n_episodes):
        model.init_state(1)
        recon = {'A': {'self': {}, 'other': {}}, 'B': {'self': {}, 'other': {}}}
        with torch.no_grad():
            for t in range(max(TIMESTEPS) + 1):
                sv_float = sv_all[e, t].astype(np.float32) / 255.0
                sv_scaled = util.scale_vision(sv_float)
                sv_t = torch.tensor(sv_scaled).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                sm_t = torch.tensor(sm_all[e, t]).unsqueeze(0).float().to(DEVICE)

                p_mask = 0.0 if t == 0 else p_mask_vision
                x = {'self_vision': sv_t, 'self_motion': sm_t}
                pred = model(x, p_mask_vision_self=p_mask, p_mask_vision_other=p_mask)

                if t in TIMESTEPS:
                    state = model.get_state()
                    ss = state['self'][0]
                    os_ = state['other'][0]

                    # Method A: FPM -> decoder directly (bypasses integration_module)
                    dec_self_a = model.vision_decoder_module(model.predict_feature(ss))
                    dec_other_a = model.vision_decoder_module(model.predict_feature(os_))
                    recon['A']['self'][t] = to_img(dec_self_a.squeeze(0))
                    recon['A']['other'][t] = to_img(dec_other_a.squeeze(0))

                    # Method B: integration_module(h, zeros) -> decoder
                    so_self_only = model.integration_module(ss, torch.zeros_like(os_))
                    so_other_only = model.integration_module(torch.zeros_like(ss), os_)
                    dec_self_b = model.vision_decoder_module(so_self_only)
                    dec_other_b = model.vision_decoder_module(so_other_only)
                    recon['B']['self'][t] = to_img(dec_self_b.squeeze(0))
                    recon['B']['other'][t] = to_img(dec_other_b.squeeze(0))

        fig, axes = plt.subplots(len(TIMESTEPS), 6, figsize=(24, 2.2 * len(TIMESTEPS)))
        col_titles = ['A-1 truth', 'h1 (method A)', 'h1 (method B)',
                      'A-2 truth', 'h2 (method A)', 'h2 (method B)']
        for row, t in enumerate(TIMESTEPS):
            truth_self = sv_all[e, t].astype(np.float32) / 255.0
            truth_other = ov_all[e, t].astype(np.float32) / 255.0

            axes[row, 0].imshow(truth_self)
            axes[row, 1].imshow(recon['A']['self'][t])
            axes[row, 2].imshow(recon['B']['self'][t])
            axes[row, 3].imshow(truth_other)
            axes[row, 4].imshow(recon['A']['other'][t])
            axes[row, 5].imshow(recon['B']['other'][t])
            for c in range(6):
                axes[row, c].set_xticks([])
                axes[row, c].set_yticks([])
            axes[row, 0].set_ylabel(f't={t}', fontsize=10)
            if row == 0:
                for c in range(6):
                    axes[row, c].set_title(col_titles[c], fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(SAVE_DIR, f'{label}_prediction_viz_ep{e}.png')
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
