"""
*** DEPRECATED / DISCARDED (2026-08-15) -- DO NOT REUSE OR RE-RUN. ***
Fig.5 is already reproduced by the ORIGINAL PAPER REPO's own pipeline:
config/exp/exp2.yml (model=Autoencoder, weight_decay=3.0) evaluated on the
`grid` dataset via the root-level analyze/analyze_vpt.py (invoked by
run_analysis_exp2.sh). Precomputed numbers live at
data/result/exp2/0/test/grid/save/vpt/1/histogram/result.txt and DO
confirm the paper's claim (other_other=0.1125 < other_self=0.1465) --
this file's ad-hoc from-scratch attempt failed to reproduce that only
because it used the wrong model class, no weight_decay regularization,
and a natural-trajectory dataset instead of the exhaustive grid. Kept in
git as a record of a failed independent-reproduction attempt, not as
usable code. If a Fig.5-style check is needed against a NEW checkpoint
(e.g. R3-A), adapt analyze_vpt.py / the Autoencoder+grid-dataset approach
instead of resurrecting this script. See docs/known_confounds.md ("Fig.5
(viewpoint-taking) reproduction: corrected after root-repo audit").
***

v4 Fig.5-equivalent perspective-taking check (2026-08-15).

Reproduces the ORIGINAL PAPER's actual decoding method (Fig.5a/b, main
text under "Perspective-taking by decoding"), which is NOT what the
earlier visualize_r3a_predictions.py did:

    "we first trained an additional visual decoder network (Visual
    Decoder) to reconstruct the input visual sensation from the encoded
    feature vectors of process-1 ... the trained Visual Decoder can also
    be used to decode the visual features of process-2"

I.e. a decoder trained ONLY on Visual Encoder-1's output (never touching
Shared Module, Integration, or FPM), then reused as-is on Visual
Encoder-2's output (both encoders see the SAME self_vision frame -- only
A-1's own camera exists in this codebase's non-R2 datasets). This is a
brand-new, independently-trained module:
  - does NOT modify or share weights with the existing
    vision_decoder_module (a separate instance, own random init)
  - both vision encoders are loaded from an existing checkpoint and
    FROZEN; only this new decoder's parameters are trained
  - trained with a plain per-frame L1 reconstruction loss (autoencoder
    style), no recurrence / Shared Module / FPM / Integration involved
    anywhere in this script

Verification order (per 2026-08-15 instruction): run against exp1_l1
first (known h2->other=0.9725) and inspect whether process-2's decoded
image actually resembles A-2's real appearance/perspective before this
method is trusted enough to apply to R3-A.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/train_fig5_decoder.py --exp_config exp1_l1 --epoch 200 \
        --train_epochs 15 --label exp1_l1
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import model as models  # noqa
from model.modules import VisionDecoderModule  # noqa
import util  # noqa

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Ground truth for the qualitative/quantitative comparison (needs real
# other_vision, which only R2's collection saved).
EVAL_DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
SAVE_DIR = 'data/result/baseline_v4'
TIMESTEPS = [10, 30, 50, 70, 90]
SEED = 12345
BATCH_SIZE = 10  # matches the paper's Methods (batch=10)


class Args:
    def __init__(self, exp_config, seed):
        self.exp_config = exp_config
        self.seed = seed


def to_img(v_scaled_m1_1):
    v = ((v_scaled_m1_1 + 1) / 2).clamp(0, 1)
    v = v.permute(1, 2, 0).detach().cpu().numpy()
    return v


def load_frozen_encoders(exp_config_name, epoch):
    exp_config = util.gen_exp_config(Args(exp_config_name, 0))
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(Args(exp_config_name, 0), test=False)

    full_model = getattr(models, exp_config.model.name)(model_config)
    full_model.to(DEVICE)
    util.load_model(model_dir, epoch, full_model)
    full_model.eval()

    enc1 = full_model.self_vision_encoder_module
    enc2 = full_model.other_vision_encoder_module
    for p in enc1.parameters():
        p.requires_grad = False
    for p in enc2.parameters():
        p.requires_grad = False
    return enc1, enc2, model_config


def load_all_frames(data_h5, split='train'):
    with h5py.File(data_h5, 'r') as f:
        # dtype varies by dataset vintage: newer r2-style collections store
        # uint8 (0-255); older collections (e.g. self_random_other_stay)
        # store float32 already scaled to [0,1]. Keep native dtype here and
        # let util.scale_vision's own dtype check handle the conversion --
        # do NOT divide by 255 in this function or the caller, or float32
        # [0,1] data gets silently divided again and crushed near 0.
        sv = f[f'{split}/self_vision'][()]  # (N, T, H, W, 3)
    return sv.reshape(-1, *sv.shape[2:])  # (N*T, H, W, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', default='exp1_l1')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--train_data', default='self_random_other_stay',
                         help='dataset to train the decoder on (paper: agent-1 self_vision '
                              'from the SAME data the base model itself trained on)')
    parser.add_argument('--label', default=None)
    parser.add_argument('--n_episodes', type=int, default=3)
    args = parser.parse_args()
    label = args.label or args.exp_config

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    enc1, enc2, model_config = load_frozen_encoders(args.exp_config, args.epoch)

    convolved_shape = enc1.get_convolved_shape()
    fig5_decoder = VisionDecoderModule(model_config.vision_decoder_module, convolved_shape)
    fig5_decoder.to(DEVICE)
    fig5_decoder.train()

    optimizer = torch.optim.AdamW(fig5_decoder.parameters(), lr=1e-3, weight_decay=0.01)

    train_data_h5 = (f'/work/my_research/preference_inference/data/data/'
                      f'{args.train_data}/data.h5')
    frames = load_all_frames(train_data_h5)
    n_frames = len(frames)
    print(f'Training Fig.5 decoder on top of frozen {args.exp_config} encoders, '
          f'data={args.train_data} (n_frames={n_frames}), '
          f'{args.train_epochs} epochs x batch={BATCH_SIZE} '
          f'({args.train_epochs * (n_frames // BATCH_SIZE)} total steps)')

    for epoch in range(args.train_epochs):
        order = rng.permutation(n_frames)
        total_loss = 0.0
        n_batches = n_frames // BATCH_SIZE
        for b in range(n_batches):
            batch_idx = np.sort(order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE])
            batch_raw = frames[batch_idx]
            batch_scaled = util.scale_vision(batch_raw)
            v = torch.tensor(batch_scaled).permute(0, 3, 1, 2).float().to(DEVICE)

            with torch.no_grad():
                enc = enc1(v)
            recon = fig5_decoder(enc)
            loss = F.l1_loss(recon, v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'  epoch {epoch+1}/{args.train_epochs}  l1_loss={total_loss / n_batches:.4f}')

    fig5_decoder.eval()

    # Save the trained decoder (for reuse / record) -- independent file, not touching any checkpoint
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(fig5_decoder.state_dict(),
               os.path.join(SAVE_DIR, f'{label}_fig5_decoder.pth'))

    # --- visualize: decoder(enc1(v)) vs decoder(enc2(v)) vs ground truth ---
    with h5py.File(EVAL_DATA_H5, 'r') as f:
        n_data = f['train/self_vision'].shape[0]
        ep_indices = rng.choice(n_data, size=args.n_episodes, replace=False)
        sv_all = np.stack([f['train/self_vision'][int(i)] for i in ep_indices])
        ov_all = np.stack([f['train/other_vision'][int(i)] for i in ep_indices])

    for e in range(args.n_episodes):
        recon_self = {}
        recon_other = {}
        with torch.no_grad():
            for t in TIMESTEPS:
                sv_float = sv_all[e, t].astype(np.float32) / 255.0
                sv_scaled = util.scale_vision(sv_float)
                v = torch.tensor(sv_scaled).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

                e1 = enc1(v)
                e2 = enc2(v)
                dec_self = fig5_decoder(e1)
                dec_other = fig5_decoder(e2)
                recon_self[t] = to_img(dec_self.squeeze(0))
                recon_other[t] = to_img(dec_other.squeeze(0))

        fig, axes = plt.subplots(len(TIMESTEPS), 4, figsize=(16, 2.2 * len(TIMESTEPS)))
        col_titles = ['A-1 truth', 'Fig5-decoder(Encoder-1)', 'A-2 truth', 'Fig5-decoder(Encoder-2)']
        for row, t in enumerate(TIMESTEPS):
            truth_self = sv_all[e, t].astype(np.float32) / 255.0
            truth_other = ov_all[e, t].astype(np.float32) / 255.0
            axes[row, 0].imshow(truth_self)
            axes[row, 1].imshow(recon_self[t])
            axes[row, 2].imshow(truth_other)
            axes[row, 3].imshow(recon_other[t])
            for c in range(4):
                axes[row, c].set_xticks([])
                axes[row, c].set_yticks([])
            axes[row, 0].set_ylabel(f't={t}', fontsize=10)
            if row == 0:
                for c in range(4):
                    axes[row, c].set_title(col_titles[c], fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(SAVE_DIR, f'{label}_fig5_decoder_viz_ep{e}.png')
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
