"""
v7 stage 2: extract A-2's viewpoint v̂² from A-1's OWN vision via the Fig.5
path, Encoder-2 -> Decoder (docs/v7_experiment_log.md §2.3, §2.8).

    (a) main:       fig5_v4_exp1l1000 ep100  (Encoder = paper exp1_l1_1000 ep200)
    (b) comparison: fig5_v6_base ep100       (Encoder = v6_s3_base_l1 ep200)

Input: test-split self_vision only (A-1's first-person vision). No A-2 data
enters the extraction. Output: v̂² in [0,1] (float16) for every test frame,
plus pre-flight checks:
  - the ep100 checkpoint's encoder tensors equal the pretrain source
    (torch.equal), so the Fig.5 Encoder-2 is the one we think it is
  - Fig.5-style errors on this data: |v̂² - other_vision|, |v̂² - self_vision|,
    and the decoder's own self reconstruction |dec(enc1(sv)) - sv|
  - a montage of true A-2 vision / v̂²(a) / v̂²(b) / A-1 vision (viewed by eye)

Usage (inside Docker, from /work/my_research/preference_inference -- util's
paths are relative to it):
    python analyze/extract_vhat2_v7.py --cond a
    python analyze/extract_vhat2_v7.py --cond b
    python analyze/extract_vhat2_v7.py --montage
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

_PI_DIR = '/work/my_research/preference_inference'
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)
sys.path.insert(0, os.path.join(_PI_DIR, 'analyze'))

import model as models  # noqa
import util  # noqa
import irl_common_v7 as C  # noqa

DATA_PATH = os.path.join(_PI_DIR, 'data', 'data', 'v7_a1random_a2goal3', 'data.h5')
SAVE_DIR = os.path.join(_PI_DIR, 'data', 'result', 'v7_irl', 'stage2')
CONDS = {
    'a': dict(exp_config='fig5_v4_exp1l1000', epoch=100, pretrain=('exp1_l1_1000', 200)),
    'b': dict(exp_config='fig5_v6_base', epoch=100, pretrain=('v6_s3_base_l1', 200)),
}
ENC_PREFIXES = ('self_vision_encoder_module.', 'other_vision_encoder_module.', 'share_lns.')


class Args:
    def __init__(self, exp_config, seed=0):
        self.exp_config = exp_config
        self.seed = seed


def ckpt_path(exp_config, epoch):
    return os.path.join(_PI_DIR, 'data', 'result', exp_config, '0', 'model', '{:05d}.pth'.format(epoch))


def load_autoencoder(cond, device):
    spec = CONDS[cond]
    exp_config = util.gen_exp_config(Args(spec['exp_config']))
    net = getattr(models, exp_config.model.name)(util.gen_model_config(exp_config))
    state = torch.load(ckpt_path(spec['exp_config'], spec['epoch']), map_location='cpu')['model']
    net.load_state_dict(state, strict=True)
    ref = torch.load(ckpt_path(*spec['pretrain']), map_location='cpu')['model']
    keys = sorted(k for k in state if k.startswith(ENC_PREFIXES))
    n_eq = sum(1 for k in keys if k in ref and torch.equal(state[k], ref[k]))
    if n_eq != len(keys) or len(keys) == 0:
        raise RuntimeError(f'({cond}) encoder tensors: {n_eq}/{len(keys)} equal to pretrain source')
    return net.to(device).eval(), dict(n_encoder_tensors=len(keys), n_equal_to_pretrain=n_eq,
                                       checkpoint=ckpt_path(spec['exp_config'], spec['epoch']),
                                       pretrain=ckpt_path(*spec['pretrain']))


@torch.no_grad()
def extract(net, sv_u8, device, batch=2048):
    """sv_u8 (M, H, W, 3) uint8 -> v̂² and self reconstruction, (M, H, W, 3) float32 in [0,1]."""
    vhat, srec = [], []
    for i0 in range(0, len(sv_u8), batch):
        x = sv_u8[i0:i0 + batch].astype(np.float32) / 255.0 * 2 - 1
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)
        rec = net(x, decode_from_other=True)
        for out, key in ((vhat, 'other_vision'), (srec, 'self_vision')):
            y = ((rec[key] + 1) / 2).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
            out.append(y)
    return np.concatenate(vhat), np.concatenate(srec)


def run_cond(cond):
    C.set_determinism(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f = h5py.File(DATA_PATH, 'r')
    sv = f['test/self_vision'][()]
    ov = f['test/other_vision'][()]
    N, T = sv.shape[:2]
    net, check = load_autoencoder(cond, device)
    print(f'({cond}) encoder check: {check["n_equal_to_pretrain"]}/{check["n_encoder_tensors"]} equal')
    vhat, srec = extract(net, sv.reshape(N * T, *sv.shape[2:]), device)
    sv01 = sv.reshape(N * T, *sv.shape[2:]).astype(np.float32) / 255.0
    ov01 = ov.reshape(N * T, *ov.shape[2:]).astype(np.float32) / 255.0
    errs = dict(
        other_other=float(np.abs(vhat - ov01).mean()),   # v̂² vs A-2's true vision
        other_self=float(np.abs(vhat - sv01).mean()),    # v̂² vs A-1's own vision
        self_self=float(np.abs(srec - sv01).mean()),     # decoder's training objective
        self_other=float(np.abs(srec - ov01).mean()),
    )
    print(f'({cond}) Fig.5-style errors on v7 test: {errs}')
    os.makedirs(SAVE_DIR, exist_ok=True)
    out = os.path.join(SAVE_DIR, f'vhat2_{cond}_test.h5')
    with h5py.File(out, 'w') as h:
        h.create_dataset('vhat2', data=vhat.reshape(N, T, *vhat.shape[1:]).astype(np.float16), compression='gzip')
        h.attrs['cond'] = cond
        h.attrs['checkpoint'] = check['checkpoint']
    meta = dict(cond=cond, spec=CONDS[cond], encoder_check=check, fig5_errors_test=errs,
                dataset='v7_a1random_a2goal3', split='test', input='A-1 self_vision only',
                vhat2_file=out, dtype='float16')
    with open(os.path.join(SAVE_DIR, f'vhat2_{cond}_meta.json'), 'w') as fp:
        json.dump(meta, fp, indent=1)
    print(f'Saved: {out}')


def montage():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = h5py.File(DATA_PATH, 'r')
    sv, ov = f['test/self_vision'], f['test/other_vision']
    goal = f['test/goal_index'][()]
    opos = f['test/other_position'][()]
    near = np.linalg.norm(opos - C.GOAL_POS[goal][:, None], axis=-1) < C.NEAR_DIST
    va = h5py.File(os.path.join(SAVE_DIR, 'vhat2_a_test.h5'), 'r')['vhat2']
    vb = h5py.File(os.path.join(SAVE_DIR, 'vhat2_b_test.h5'), 'r')['vhat2']
    rng = np.random.RandomState(0)
    picks = []
    for g in range(C.N_MAIN):
        for want_near in (False, True):
            cand = np.argwhere(near & (goal[:, None] == g) if want_near else (~near) & (goal[:, None] == g))
            n, t = cand[rng.randint(len(cand))]
            picks.append((g, want_near, int(n), int(t)))
    fig, ax = plt.subplots(len(picks), 4, figsize=(14, 1.15 * len(picks) + 0.6))
    titles = ['A-2 true vision', 'v̂² (a) paper Enc', 'v̂² (b) v6 Enc', 'A-1 vision (input)']
    for i, (g, nr, n, t) in enumerate(picks):
        ims = [ov[n, t] / 255.0, np.asarray(va[n, t], np.float32), np.asarray(vb[n, t], np.float32), sv[n, t] / 255.0]
        for j, im in enumerate(ims):
            ax[i, j].imshow(np.clip(im, 0, 1), aspect='auto')
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(titles[j], fontsize=9)
        ax[i, 0].set_ylabel(f'{C.CAND_NAMES[g]} {"<5" if nr else "≥5"}\nep{n} t{t}', fontsize=7)
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, 'vhat2_montage.png')
    fig.savefig(out, dpi=120)
    print(f'Saved: {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', choices=list(CONDS))
    ap.add_argument('--montage', action='store_true')
    args = ap.parse_args()
    if args.cond:
        run_cond(args.cond)
    if args.montage:
        montage()


if __name__ == '__main__':
    main()
