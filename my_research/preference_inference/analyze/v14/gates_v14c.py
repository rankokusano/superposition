"""
v14-C stage-1 gates (docs/v14_experiment_log.md Sec.31, fixed before computing). Runs the v12 measures with "Red"
replaced by the episode's liked colour L (disliked D = the other of Red / Green), without changing v12 / v11 code:
  * every model (v14-C experienced, v12 no-experience, v14-C no-experience version) is loaded into
    model_v14.SuperpositionNetworkMotionPleasureHoldSigned (same weight keys as the v12 class) and fed x['a1_liked_t']
    (1 Red / 0 Green) per episode by LikedFeed, so all conditions get the same signed pleasure input;
  * common test = v14_s1_randompref test (= v11_s1_random test + a1_liked);
  * s1 mode (mse late-5): eval_trial_v12.run / freeze, eval_m1_v11.m1c / m1d (per-colour values), localise_v11 d*
    slopes -> per episode c_pred_L, c_cf_L, colour S_L - S_D, g_L; tracking R^2 (visible) overall and per liked colour;
  * g5 mode (l1 late-5): eval_g5_v12.g5a_est / g5a_metric with A-1's liked colour held constant (Red main, Green
    alongside), and G5b (the v12 g5b loop, copied here only to pass the per-episode liked colour; g5b_summary imported).
Per-episode values -> data/result/v14/gates/s<seed>/<mode>_<cond>_ep<e><tag>.npz (+ .json); judgement:
aggregate_gates_v14c.py.
Usage: python analyze/v14/gates_v14c.py s1|g5 --seed s --conds exp v12noexp [noexpv] [--epochs ...] [--tag _rep]
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, _PI)
import eval_trial_v12 as T  # noqa
import eval_g5_v12 as G  # noqa
from eval_m1_v11 import m1c, m1d  # noqa
from localise_v11 import dstar, slopes  # noqa
from p2test_v11 import calibrate, read_a2  # noqa
from config_util import load_config  # noqa
from model import model_v14  # noqa
import util  # noqa

COMMON = os.path.join(_PI, 'data', 'data', 'v14_s1_randompref', 'data.h5')
T.COMMON = COMMON
G.COMMON = COMMON
OUT = os.path.join(_PI, 'data', 'result', 'v14', 'gates')
COLORS = ('Red', 'Green', 'Blue', 'Cyan')
EXPS = {'exp': ('v14_s1_mixedpref_mse', 'v14_s1_mixedpref_l1_s{s}'),
        'v12noexp': ('v12_s1_random_mse', 'v12_s1_random_l1_s{s}'),
        'noexpv': ('v14_s1_randompref_mse', 'v14_s1_randompref_l1_s{s}')}


def load_signed(exp, seed, epoch, dev):
    cfg = load_config(os.path.join(_PI, 'config', 'exp', exp + '.yml'))
    assert cfg.model.name in ('SuperpositionNetworkMotionPleasureHold', 'SuperpositionNetworkMotionPleasureHoldSigned')
    mcfg = load_config(os.path.join(_PI, 'config', 'model', cfg.model.name, cfg.model.config + '.yml'))
    m = model_v14.SuperpositionNetworkMotionPleasureHoldSigned(mcfg).to(dev)
    m.load_state_dict(torch.load(os.path.join(_PI, 'data', 'result', exp, str(seed), 'model', f'{epoch:05d}.pth'),
                                 map_location='cpu')['model'])
    m.eval()
    pm = cfg.get('p_mask_vision', None)
    return m, (None if pm is None else float(pm))


class LikedFeed:
    """Passes x['a1_liked_t'] for the current batch: per episode (set_episodes, batches in order, one init_state per
    batch) or a constant (set_const)."""

    def __init__(self, model):
        for k, v in (('model', model), ('arr', None), ('const', None), ('b0', 0), ('next0', 0), ('B', 0)):
            object.__setattr__(self, k, v)

    def set_episodes(self, arr):
        object.__setattr__(self, 'arr', np.asarray(arr)); object.__setattr__(self, 'const', None)
        object.__setattr__(self, 'next0', 0)

    def set_const(self, v):
        object.__setattr__(self, 'const', int(v)); object.__setattr__(self, 'arr', None)

    def set_batch(self, b0, B):
        object.__setattr__(self, 'b0', b0); object.__setattr__(self, 'next0', b0 + B); object.__setattr__(self, 'B', B)

    def init_state(self, B):
        object.__setattr__(self, 'b0', self.next0); object.__setattr__(self, 'next0', self.next0 + B)
        object.__setattr__(self, 'B', B)
        self.model.init_state(B)

    def __call__(self, x, *a, **k):
        x = dict(x)
        dev = x['self_vision'].device
        B = x['self_vision'].shape[0]
        lk = np.full(B, self.const) if self.const is not None else self.arr[self.b0:self.b0 + B]
        assert len(lk) == B
        x['a1_liked_t'] = torch.from_numpy(np.asarray(lk, np.int64)).to(dev)
        return self.model(x, *a, **k)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        setattr(self.model, name, value)


def by_liked(v, L, nan_mean):
    """v: dict colour -> (N,) ; L: (N,) liked index (0 Red, 1 Green) -> (v_L - mean(other 3), v_L - v_D).
    As v12: M1(c) / M1(d) use the plain mean of the other three (NaN if any is NaN), g uses nanmean (g_of)."""
    A = np.stack([v[X] for X in COLORS], 1)
    n = np.arange(len(L))
    others = np.stack([np.delete(A[i], L[i]) for i in n])
    return A[n, L] - (np.nanmean(others, 1) if nan_mean else others.mean(1)), A[n, L] - A[n, 1 - L]


def tracking_split(d, liked):
    N = d['h'].shape[0]
    X = d['h'].reshape(-1, 128)
    Y = d['pos'][:, 1:101].reshape(-1, 2)
    grp = np.repeat(np.arange(N)[:, None], 100, 1).reshape(-1)
    hid = d['hidden'].reshape(-1)
    lk = np.repeat(liked[:, None], 100, 1).reshape(-1)
    Yh = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(X, Y, grp):
        Yh[te] = Ridge(alpha=1.0).fit(X[tr], Y[tr]).predict(X[te])

    def r2(m):  # noqa
        return float(1 - ((Y[m] - Yh[m]) ** 2).sum() / ((Y[m] - Y[m].mean(0)) ** 2).sum())
    out = {}
    for nm, mk in (('all', np.ones_like(hid)), ('red', lk == 1), ('green', lk == 0)):
        out[nm] = {'all': r2(mk), 'visible': r2(mk & ~hid), 'hidden': r2(mk & hid)}
    return out


def s1(model, p_mask, dev, liked, cf, cfv, views, vvalid):
    L = np.where(liked == 1, 0, 1)          # liked colour index in COLORS (Red 0, Green 1)
    model.set_episodes(liked)
    d = T.run(model, p_mask, dev)
    r = {'tracking': tracking_split(d, liked)}
    per = {'liked': liked, 'layout': d['layout']}
    for nm, fm in (('hidden', d['hidden']), ('visible', ~d['hidden'])):
        delta, _ = m1c(d, d['src'], fm)
        S, _ = m1d(d, d['src'], cf, cfv, fm)
        per[f'{nm}_c_pred'], _ = by_liked(delta, L, False)
        per[f'{nm}_c_cf'], ld = by_liked(S, L, False)
        per[f'{nm}_colour_all'] = ld
        per[f'{nm}_colour_swap'] = np.where(d['layout'] == 1, ld, np.nan)
        per[f'{nm}_c_pred_red_def'] = delta['Red'] - (delta['Green'] + delta['Blue'] + delta['Cyan']) / 3   # v12 definition (check)
    del d
    for K, variant, hold in ((15, 'hold', True), (20, 'hold', True), (15, 'zero', False)):
        model.set_episodes(liked)
        pr = T.freeze(model, p_mask, dev, K, hold)
        sl = slopes(dstar(pr, views, vvalid), K)
        per[f'g_K{K}_{variant}'], _ = by_liked({X: sl[:, i] for i, X in enumerate(COLORS)}, L, True)
        per[f'slope_K{K}_{variant}'] = sl
    return r, per


@torch.no_grad()
def g5b_liked(model, p_mask, dev, liked, K=10, batch=50):
    """eval_g5_v12.g5b with A-1's liked colour per episode (the loop copied; summary imported)."""
    g = h5py.File(COMMON, 'r')['test']
    sp_all, op_all = g['self_position'][()], g['other_position'][()]
    alpha, beta = calibrate(g['self_vision'][:, :100:5], sp_all[:, :100:5], op_all[:, :100:5])
    N = sp_all.shape[0]
    est = {}
    model.set_episodes(liked)
    for b0 in range(0, N, batch):
        x, m = T.batch_x(g, b0, batch, dev)
        B = x.shape[0]
        sp = sp_all[b0:b0 + batch, G.T0:G.T0 + K]
        keys = [('ctrl', None)] + [(f'{sg}{k}', (sgn, k)) for sg, sgn in (('p', 1.0), ('m', -1.0)) for k in range(8)]
        for key, spec in keys:
            Q = torch.zeros(B, 11, device=dev)
            Q[:, 10] = 1.0
            if spec is not None:
                Q[:, 2 + spec[1]] = spec[0] * G.A
            util.seed_all(0)
            model.set_batch(b0, 0)
            model.init_state(B)
            prs = []
            for t in range(G.T0 + K):
                pm = 0 if t == 0 else p_mask
                pr = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm,
                           p2_override=Q if t >= G.T0 else None)['self_vision']
                if t >= G.T0:
                    prs.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
            est.setdefault(key, []).append(np.stack([read_a2(prs[i], sp[:, i], alpha, beta) for i in range(K)], 1))
    est = {k: np.concatenate(v) for k, v in est.items()}
    res, main = G.g5b_summary(est)
    return res, main


def g5(model, p_mask, dev, liked, D, PA2, ab):
    r, per = {}, {}
    model.motion_masking = False              # stage-2 protocol for G5a
    for a1 in (1, 0):                          # A-1 liked Red (main) / Green (alongside)
        model.set_const(a1)
        for k in D:
            for K in (15, 20):
                est = G.g5a_est(model, D[k], PA2[k], dev, ab[0], ab[1], K, 'freeze')
                gg, sl, S = G.g5a_metric(est, D[k])
                tag = f'a1{"red" if a1 else "green"}_{k}_K{K}'
                per[f'G5a_{tag}'] = gg
                r[f'G5a_{tag}_frac_read'] = float(np.isfinite(est[..., 0]).mean())
    model.motion_masking = True               # canonical for G5b (as v12)
    rb, main_b = g5b_liked(model, p_mask, dev, liked)
    r['G5b'] = rb
    per['G5b'] = main_b
    per['liked'] = liked
    per['a2_liked_pref'] = D['pref']['a2_liked']
    return r, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['s1', 'g5'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--conds', nargs='+', default=['exp', 'v12noexp'])
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    out = os.path.join(OUT, f's{a.seed}')
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    liked = h5py.File(COMMON, 'r')['test']['a1_liked'][()].astype(int)
    if a.mode == 's1':
        fcf = h5py.File(os.path.join(T.RES, 'v11', 'cf', 'cf_views.h5'), 'r')
        cf, cfv = fcf['cf_vision'][()], fcf['valid'][()]
        fv = h5py.File(os.path.join(T.RES, 'v11', 'cf', 'cf_dist_views.h5'), 'r')
        views, vvalid = fv['views'][()], fv['valid'][()]
    else:
        D = {k: G.load_set(k) for k in ('green', 'pref')}
        PA2 = {k: G.true_pleasure(k, D[k]) for k in D}
        gg = h5py.File(D['green']['path'], 'r')['test']
        ab = calibrate(gg['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    for c in a.conds:
        for e in a.epochs:
            exp = EXPS[c][0 if a.mode == 's1' else 1].format(s=a.seed)
            m, pm = load_signed(exp, a.seed, e, dev)
            model = LikedFeed(m)
            if a.mode == 's1':
                r, per = s1(model, pm, dev, liked, cf, cfv, views, vvalid)
            else:
                r, per = g5(model, pm, dev, liked, D, PA2, ab)
            r.update(exp=exp, seed=a.seed, epoch=e, cond=c)
            fn = os.path.join(out, f'{a.mode}_{c}_ep{e}{a.tag}')
            json.dump(r, open(fn + '.json', 'w'), indent=1)
            np.savez_compressed(fn + '.npz', **per)
            show = {k: round(float(np.nanmean(v)), 4) for k, v in per.items() if k not in ('liked', 'layout', 'a2_liked_pref')
                    and np.asarray(v).ndim == 1 and not k.startswith('slope')}
            print(a.mode, c, e, json.dumps(show), flush=True)


if __name__ == '__main__':
    main()
