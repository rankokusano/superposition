"""
Encoder differentiation check (2026-09-08, user request during v5 review).

Question: in v5 the base-stage h2->self R^2 jumped to ~0.23 (vs ~0.06 for
R3-stay / R3-A). Candidate explanation: the from-scratch Encoder-2
(other_vision_encoder_module) converged to a representation close to
Encoder-1 (self_vision_encoder_module), so process-2 receives essentially
process-1 features and self/other separation collapses at the Encoder level.

Test: push the SAME frames through both encoders and measure how similar the
two outputs are. High similarity => the two encoders are (near) the same
function => no Encoder-level self/other differentiation.

Compares:
  - v5_base_mse   ep400   (Encoder trained from scratch under probe-Q input)
  - r3_stay_400/0 ep400   (Encoder frozen from motion regime = exp1_l1_1000)
  - exp1_l1_1000/0 ep200  (the original paper's Encoder -- gold reference)

Usage (from /work/my_research/preference_inference, NO CUDA_VISIBLE_DEVICES clamp):
  DEVICE=cuda:6 python analyze/encoder_similarity.py
"""
import os
import sys

import h5py
import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if '/work' not in sys.path:
    sys.path.insert(0, '/work')
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import model as models  # noqa
import util  # noqa

DEVICE = torch.device(os.environ.get('DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
T = 100
N_EP = 200          # episodes
BATCH = 100


class Args:
    def __init__(self, exp_config, seed):
        self.exp_config = exp_config
        self.seed = seed


def load_net(exp_config, epoch, seed=0):
    ec = util.gen_exp_config(Args(exp_config, seed))
    mc = util.gen_model_config(ec)
    _, model_dir, _ = util.gen_dirs(Args(exp_config, seed), test=False)
    net = getattr(models, ec.model.name)(mc)
    net.to(DEVICE)
    util.load_model(model_dir, epoch, net)
    net.eval()
    return net


def frames_from(h5_path, key, n_ep=N_EP):
    with h5py.File(h5_path, 'r') as f:
        arr = f[f'train/{key}'][:n_ep, :T]          # (n, T, H, W, C) uint8
    arr = arr.reshape(-1, *arr.shape[2:]).astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr                                       # (n*T, H, W, C) in [0,1]


def encode(net, frames):
    """Run both encoders on the same frames; return (sv_enc, ov_enc) as np."""
    sv, ov = [], []
    with torch.no_grad():
        for b0 in range(0, len(frames), BATCH):
            fb = frames[b0:b0 + BATCH]
            t = torch.tensor(util.scale_vision(fb)).permute(0, 3, 1, 2).float().to(DEVICE)
            sv.append(net.self_vision_encoder_module(t).cpu().numpy())
            ov.append(net.other_vision_encoder_module(t).cpu().numpy())
    return np.concatenate(sv), np.concatenate(ov)


def report(name, sv, ov):
    # per-sample cosine similarity between the two encoders' outputs
    a = sv / (np.linalg.norm(sv, axis=1, keepdims=True) + 1e-8)
    b = ov / (np.linalg.norm(ov, axis=1, keepdims=True) + 1e-8)
    cos = (a * b).sum(axis=1)
    # per-dimension Pearson r across samples
    def _z(x):
        return (x - x.mean(0)) / (x.std(0) + 1e-8)
    r = (_z(sv) * _z(ov)).mean(0)
    # variance explained if we regress ov on sv linearly (dimwise proxy: mean r^2)
    print(f'{name:28s}  cos_sim mean={cos.mean():+.3f} sd={cos.std():.3f}  '
          f'|per-dim r| mean={np.abs(r).mean():.3f}  '
          f'frac|r|>0.8={np.mean(np.abs(r) > 0.8)*100:.0f}%  '
          f'(sv‖ mean={np.linalg.norm(sv,axis=1).mean():.2f} '
          f'ov‖ mean={np.linalg.norm(ov,axis=1).mean():.2f})')


def main():
    r3 = 'data/data/r3_stay/data.h5'
    print(f'device={DEVICE}  frames: {N_EP} ep x {T} t  from r3_stay (self_vision & other_vision)\n')
    sv_frames = frames_from(r3, 'self_vision')
    ov_frames = frames_from(r3, 'other_vision')

    for label, cfg, ep in [
        ('v6_s2_base_mse ep400', 'v6_s2_base_mse', 400),
        ('v6_s3_base_l1 ep200', 'v6_s3_base_l1', 200),
        ('v5_base_mse ep400', 'v5_base_mse', 400),
        ('r3_stay_400/0 ep400 (v4)', 'r3_stay_400', 400),
        ('exp1_l1_1000/0 ep200 (paper)', 'exp1_l1_1000', 200),
    ]:
        try:
            net = load_net(cfg, ep)
        except Exception as e:
            print(f'{label}: load failed -- {e}')
            continue
        print(f'== {label} ==')
        s1, o1 = encode(net, sv_frames)
        report('  on A-1 (self) frames', s1, o1)
        s2, o2 = encode(net, ov_frames)
        report('  on A-2 (other) frames', s2, o2)
        # cross: does ov_enc(other) look like sv_enc(other)? (the collapse case)
        print()


if __name__ == '__main__':
    main()
