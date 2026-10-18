"""
v11 localisation test: is the red-ward estimate in the shared module's state or
in the Decoder? (docs/v11_experiment_log.md Sec.7.29-7.30; criteria confirmed)

Main (input-freeze test): common test episodes; t = 0..19 real inputs (canonical
masks, eval_seed 0), then K = 20 steps with vision masked (p = 1) and the
pleasure input set to zero. Each step's one-step prediction v_hat(k) is compared
with views rendered from p_20 moved d = 0..10 unit steps toward colour X's
reference (cf_dist_views.h5). d*_X(k) = argmin_d L1 over valid d; per episode the
OLS slope of d*_X over k = 1..K; g = slope_Red - mean(slope_G,B,C). (episode, X)
used only if d = 0..5 are all valid; episodes need all four colours.
Variants (descriptive): K = 10; pleasure held at its value at the freeze frame.
Descriptive (user's first idea): under canonical masks, per frame s_X as in M1(d)
grouped by k = steps since the last unmasked vision frame.
Writes JSON + figure under data/result/v11/main/localise/.
"""

import argparse
import copy
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import DATA, LAYNAME, RES, boot_ci, load_model, run  # noqa
from eval_m1_v11 import COLORS, refs  # noqa
import util  # noqa
import model.util as model_util  # noqa

T0, K, DMIN_VALID = 20, 20, 5
_LOG = []
_orig_mask = model_util.mask


def _logged_mask(x, p):
    m = torch.rand((x.size(0), 1), device=x.device) < p
    _LOG.append(m[:, 0].cpu().numpy())
    return torch.where(m, torch.zeros_like(x), x)


@torch.no_grad()
def freeze_preds(model, p_mask, sv_all, device, hold=False, batch=50):
    """Returns predictions (N, K, 16, 64, 3) float in [0,1]."""
    out = []
    for b0 in range(0, len(sv_all), batch):
        sv = sv_all[b0:b0 + batch]
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(sv))).to(device)
        B = x.shape[0]
        util.seed_all(0)
        model.init_state(B)
        for t in range(T0):
            pm = 0 if t == 0 else p_mask
            model({'self_vision': x[:, t]}, pm, pm)
        P = model.compute_pleasure(x[:, T0]) if hold else torch.zeros(B, 8, device=device)
        preds = []
        for k in range(K):
            pr = model({'self_vision': x[:, T0 + k]}, 1.0, 1.0, p1_override=P)['self_vision']
            preds.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
        out.append(np.stack(preds, 1))
    return np.concatenate(out)


def dstar(preds, views, valid):
    """preds (N,K,H,W,C); views (N,4,D,H,W,C) uint8; valid (N,4,D) -> d* (N,4,K) with NaN where unusable."""
    N = preds.shape[0]
    D = views.shape[2]
    out = np.full((N, 4, preds.shape[1]), np.nan)
    for n in range(N):
        for c in range(4):
            if not valid[n, c, :DMIN_VALID + 1].all():
                continue
            v = views[n, c].astype(np.float32) / 255.0                      # (D,H,W,C)
            L = np.abs(preds[n][:, None] - v[None]).mean((2, 3, 4))         # (K, D)
            L[:, ~valid[n, c]] = np.inf
            out[n, c] = L.argmin(1)
    return out


def slopes(ds, kmax):
    ks = np.arange(1, kmax + 1)
    s = np.full(ds.shape[:2], np.nan)
    for n in range(ds.shape[0]):
        for c in range(4):
            y = ds[n, c, :kmax]
            if not np.isnan(y).any():
                s[n, c] = np.polyfit(ks, y, 1)[0]
    return s


def g_of(s):
    return s[:, 0] - np.nanmean(s[:, 1:], 1) if True else None


def summ(v):
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', default='v11_s1_{}_mse')
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--sub', default='main/localise')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    out = os.path.join(RES, 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    path = os.path.join(DATA, 'v11_s1_random', 'data.h5')
    g = h5py.File(path, 'r')['test']
    sv_all = g['self_vision'][()]
    lay = g['layout'][()]
    src = g['a1_source'][()]
    fv = h5py.File(os.path.join(RES, 'v11', 'cf', 'cf_dist_views.h5'), 'r')
    views, valid = fv['views'][()], fv['valid'][()]
    fcf = h5py.File(os.path.join(RES, 'v11', 'cf', 'cf_views.h5'), 'r')
    cf1, cf1_valid = fcf['cf_vision'][()], fcf['valid'][()]
    truth = sv_all[:, 1:101]
    res, per_g, curves = {}, {}, {}
    for s in args.seeds:
        for cond in ('random', 'mixed'):
            model, p_mask = load_model(args.exp_fmt.format(cond), s, args.epoch, dev)
            r = {}
            for variant, hold in (('zero', False), ('hold', True)):
                pr = freeze_preds(model, p_mask, sv_all, dev, hold=hold)
                ds = dstar(pr, views, valid)
                for kmax in (K, 10):
                    sl = slopes(ds, kmax)
                    gg = g_of(sl)
                    key = f'{variant}_K{kmax}'
                    r[key] = {'g': summ(gg), 'slope': {X: summ(sl[:, i]) for i, X in enumerate(COLORS)},
                              'g_per_layout': {LAYNAME[k]: summ(gg[lay == k]) for k in (0, 1)}}
                    if variant == 'zero' and kmax == K:
                        per_g[(cond, s)] = gg
                        curves[(cond, s)] = {X: np.nanmean(ds[:, i], 0).tolist() for i, X in enumerate(COLORS)}
                        curves[(cond, s, 'layout')] = {LAYNAME[k]: {X: np.nanmean(ds[lay == k, i], 0).tolist()
                                                                    for i, X in enumerate(COLORS)} for k in (0, 1)}
            # descriptive: canonical run, s_X per frame grouped by steps since the last unmasked frame
            _LOG.clear()
            model_util.mask = _logged_mask
            d = run(model, p_mask, path, 'test', 0, dev, keep_pred='f16')
            model_util.mask = _orig_mask
            nb = int(np.ceil(len(sv_all) / 50))
            vis = []
            for b in range(nb):
                chunk = _LOG[b * 200:(b + 1) * 200]
                vis.append(~np.stack(chunk[0::2], 1))       # (B, 100) True = vision shown
            vis = np.concatenate(vis)
            since = np.zeros_like(vis, dtype=int)
            for t in range(100):
                since[:, t] = 0 if t == 0 else np.where(vis[:, t], 0, since[:, t - 1] + 1)
            pred = d['pred'].astype(np.float32)
            err = np.abs(pred - truth.astype(np.float32) / 255.0).mean((2, 3, 4))
            pos = d['pos']
            m = pos[:, 1:101] - pos[:, :100]
            ok0 = (src[:, :100] != 3) & (np.linalg.norm(m, axis=-1) > 1e-6)
            sx = {}
            for i, X in enumerate(COLORS):
                u = refs(lay, X) - pos[:, :100]
                dist = np.linalg.norm(u, axis=-1)
                cosang = (m * u).sum(-1) / np.maximum(np.linalg.norm(m, axis=-1) * dist, 1e-9)
                sel = ok0 & cf1_valid[:, :, i] & (cosang < np.cos(np.deg2rad(45)))
                lcf = np.abs(pred - cf1[:, :, i].astype(np.float32) / 255.0).mean((2, 3, 4))
                sxi = np.where(sel, err - lcf, np.nan)
                sx[X] = sxi
            bins = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 99)]
            r['descriptive_since_last_vision'] = {
                f'k{a}-{b}': {X: float(np.nanmean(np.where((since >= a) & (since <= b), sx[X], np.nan))) for X in COLORS}
                for a, b in bins}
            r['descriptive_vision_shown_frac_t1_99'] = float(vis[:, 1:].mean())
            res[f'{cond}_s{s}'] = r
            print(cond, s, 'g zero K20', r['zero_K20']['g'], 'slope red', r['zero_K20']['slope']['Red']['mean'], flush=True)
    paired = {}
    for s in args.seeds:
        paired[f's{s}'] = summ(per_g[('mixed', s)] - per_g[('random', s)])
    ok = {s: (res[f'mixed_s{s}']['zero_K20']['g']['mean'] > 0 and res[f'mixed_s{s}']['zero_K20']['g']['ci95'][0] > 0)
          for s in args.seeds}
    okd = {s: paired[f's{s}']['ci95'][0] > 0 for s in args.seeds}
    res['paired_mixed_minus_random_zero_K20'] = paired
    res['judgement'] = {'mixed_g_pos': {str(k): bool(v) for k, v in ok.items()},
                        'paired_pos': {str(k): bool(v) for k, v in okd.items()},
                        'pass': bool(sum(ok.values()) >= 2 and sum(okd.values()) >= 2)}
    with open(os.path.join(out, f'localise_ep{args.epoch}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    fig, axes = plt.subplots(2, len(args.seeds), figsize=(5 * len(args.seeds), 8), squeeze=False)
    cols = {'Red': 'red', 'Green': 'limegreen', 'Blue': 'blue', 'Cyan': 'c'}
    for j, s in enumerate(args.seeds):
        for i, cond in enumerate(('random', 'mixed')):
            a_ = axes[i, j]
            for X in COLORS:
                a_.plot(range(1, K + 1), curves[(cond, s)][X], color=cols[X], label=X)
            a_.set_title(f'{cond} s{s}: g = {res[f"{cond}_s{s}"]["zero_K20"]["g"]["mean"]:+.3f}', fontsize=9)
            a_.set_xlabel('frozen step k'); a_.set_ylabel('mean d* (steps toward X)')
            a_.legend(fontsize=7)
    fig.suptitle('Input-freeze test: best-matching distance toward each colour while vision and pleasure are zero '
                 f'(from t={T0}), mse ep{args.epoch}', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, f'localise_ep{args.epoch}.png'), dpi=110)
    print(json.dumps(res['judgement']), json.dumps(paired))


if __name__ == '__main__':
    main()
