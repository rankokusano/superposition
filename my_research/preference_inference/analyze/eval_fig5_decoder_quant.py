"""
*** DEPRECATED / DISCARDED (2026-08-15) -- DO NOT REUSE OR RE-RUN. ***
Evaluates the decoder trained by the also-deprecated train_fig5_decoder.py.
Fig.5 is already reproduced by the ORIGINAL PAPER REPO's own pipeline
(config/exp/exp2.yml + root-level analyze/analyze_vpt.py); precomputed
numbers live at data/result/exp2/0/test/grid/save/vpt/1/histogram/result.txt
and DO confirm the paper's claim. Kept in git as a record of a failed
independent-reproduction attempt. See train_fig5_decoder.py's header and
docs/known_confounds.md ("Fig.5 (viewpoint-taking) reproduction: corrected
after root-repo audit") for the full explanation.
***

Quantitative version of the Fig.5 check, matching the paper's Fig.5e
(bar chart of |v - decoded v| differences, not just eyeballing images).
Computes, over many real (episode, t) samples:
    err_self_self   = |decoder(enc1(v)) - true_self_vision|
    err_other_other = |decoder(enc2(v)) - true_other_vision|
    err_self_other  = |decoder(enc1(v)) - true_other_vision|   (cross term)
    err_other_self  = |decoder(enc2(v)) - true_self_vision|    (cross term)

Paper's claim corresponds to: err_other_other < err_other_self (decoded
process-2 output is closer to A-2's true vision than to A-1's), even if
not a crisp visual match. err_self_self should be the smallest of all
four (that's the decoder's actual training objective).

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/eval_fig5_decoder_quant.py --exp_config exp1_l1 --label exp1_l1
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import model as models  # noqa
from model.modules import VisionDecoderModule  # noqa
import util  # noqa

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
SAVE_DIR = 'data/result/baseline_v4'
N_SAMPLE = 3000
SEED = 999


class Args:
    def __init__(self, exp_config, seed):
        self.exp_config = exp_config
        self.seed = seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', default='exp1_l1')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--label', default=None)
    args = parser.parse_args()
    label = args.label or args.exp_config

    exp_config = util.gen_exp_config(Args(args.exp_config, 0))
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(Args(args.exp_config, 0), test=False)

    full_model = getattr(models, exp_config.model.name)(model_config)
    full_model.to(DEVICE)
    util.load_model(model_dir, args.epoch, full_model)
    full_model.eval()
    enc1 = full_model.self_vision_encoder_module
    enc2 = full_model.other_vision_encoder_module

    convolved_shape = enc1.get_convolved_shape()
    decoder = VisionDecoderModule(model_config.vision_decoder_module, convolved_shape)
    decoder.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, f'{label}_fig5_decoder.pth'), map_location=DEVICE))
    decoder.to(DEVICE)
    decoder.eval()

    rng = np.random.RandomState(SEED)
    with h5py.File(DATA_H5, 'r') as f:
        n_data, seq_len = f['train/self_vision'].shape[:2]
        flat_idx = rng.choice(n_data * seq_len, size=N_SAMPLE, replace=False)
        n_idx = flat_idx // seq_len
        t_idx = flat_idx % seq_len
        order = {}
        for i, (ne, t) in enumerate(zip(n_idx, t_idx)):
            order.setdefault(int(ne), []).append((i, int(t)))
        sv = np.zeros((N_SAMPLE, 16, 64, 3), dtype=np.uint8)
        ov = np.zeros((N_SAMPLE, 16, 64, 3), dtype=np.uint8)
        for ne, items in order.items():
            ep_sv = f['train/self_vision'][ne]
            ep_ov = f['train/other_vision'][ne]
            for i, t in items:
                sv[i] = ep_sv[t]
                ov[i] = ep_ov[t]

    errs = {'self_self': [], 'other_other': [], 'self_other': [], 'other_self': []}
    BATCH = 100
    with torch.no_grad():
        for b in range(0, N_SAMPLE, BATCH):
            sv_b = sv[b:b + BATCH].astype(np.float32) / 255.0
            ov_b = ov[b:b + BATCH].astype(np.float32) / 255.0
            sv_scaled = util.scale_vision(sv_b)
            v = torch.tensor(sv_scaled).permute(0, 3, 1, 2).float().to(DEVICE)
            true_self = torch.tensor(sv_scaled).permute(0, 3, 1, 2).float().to(DEVICE)
            true_other = torch.tensor(util.scale_vision(ov_b)).permute(0, 3, 1, 2).float().to(DEVICE)

            dec_self = decoder(enc1(v))
            dec_other = decoder(enc2(v))

            errs['self_self'].append((dec_self - true_self).abs().mean(dim=[1, 2, 3]).cpu().numpy())
            errs['other_other'].append((dec_other - true_other).abs().mean(dim=[1, 2, 3]).cpu().numpy())
            errs['self_other'].append((dec_self - true_other).abs().mean(dim=[1, 2, 3]).cpu().numpy())
            errs['other_self'].append((dec_other - true_self).abs().mean(dim=[1, 2, 3]).cpu().numpy())

    summary = {}
    for k, v in errs.items():
        arr = np.concatenate(v)
        summary[k] = {'mean': float(arr.mean()), 'std': float(arr.std())}

    ss = summary['self_self']['mean']
    oo = summary['other_other']['mean']
    so = summary['self_other']['mean']
    os_ = summary['other_self']['mean']

    print('All values are L1 reconstruction error (pixel space, [-1,1] scale) -- LOWER is better/closer.')
    print(f'  self_self   (decoder(enc1) vs true A-1): {ss:.4f}  <- reference: should be the smallest '
          f'of all four (this IS the training objective)')
    print(f'  other_other (decoder(enc2) vs true A-2): {oo:.4f}')
    print(f'  self_other  (decoder(enc1) vs true A-2): {so:.4f}')
    print(f'  other_self  (decoder(enc2) vs true A-1): {os_:.4f}')

    print(f"\n[Paper's claim] decoded process-2 output should be CLOSER to true A-2 than to true A-1, "
          f"i.e. other_other < other_self: {oo:.4f} < {os_:.4f} -> {oo < os_}")
    print(f"[Self/other confound check] is process-1's OWN decode already closer to true A-2 than "
          f"process-2's decode is (self_other < other_other)? {so:.4f} < {oo:.4f} -> {so < oo}  "
          f"(if True: process-2 is not adding other-specific information beyond what process-1 already has)")

    summary['paper_claim_other_other_lt_other_self'] = bool(oo < os_)
    summary['self_other_lt_other_other'] = bool(so < oo)

    out_path = os.path.join(SAVE_DIR, f'{label}_fig5_decoder_quant.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
