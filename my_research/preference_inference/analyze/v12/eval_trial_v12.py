"""
v12 trial evaluation (docs/v12_experiment_log.md Sec.4, registered before computing).
Common test = v11_s1_random test (300 ep). Canonical: vision mask 0.99 (t = 0 shown), motion hidden
by the model's own schedule (drawn from the CPU generator after util.seed_all(0)), model.eval(),
cudnn deterministic.
  1. self-position tracking: h1 -> p_{t+1} and p_t, Ridge(1), episode GroupKFold(5), R^2 on
     visible / hidden frames and by step within the hidden interval;
  2. M1(c), M1(d) (v11 formulas, cf_views.h5): all / motion-hidden / motion-visible frames,
     paired mixed - random;
  3. input-freeze: t < 20 canonical, then K steps motion hidden (flag 1), vision p = 1, pleasure
     held at its t = 20 value (main) or zero; d*_X slopes vs cf_dist_views.h5, g = Red - others,
     K = 15 (main) and 20;
  4. self/other code alignment cos(ov(A-1 view), sv(A-2 view)) and control cos(ov(A-1), sv(A-1))
     (original layout, every 5th step);
  5. process-2 test (ep150 only): t = 20..29 process-2 = [0, 0, +-a e_k, flag] vs control [0, 0, 0, flag];
     flag 1 (motion hidden; main, as planned for stage 2) and flag 0 (supplement);
     d_spec from the A-2 position read from the predicted image.
Writes data/result/v12/trial/eval_<cond>_ep<e>.json (+ paired file).
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
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, _PI)
from eval_trial_v11 import DATA, RES, boot_ci  # noqa
from eval_m1_v11 import m1c, m1d  # noqa
from localise_v11 import dstar, slopes, g_of  # noqa
from p2test_v11 import calibrate, read_a2  # noqa
from fig5_extra_v11 import cosm  # noqa
import util  # noqa
from config_util import load_config  # noqa
from model import model_v12  # noqa
from v11core.pleasure import BIN_UNIT  # noqa

OUT = os.path.join(RES, 'v12', 'trial')
PI_DIR_CFG = os.path.join(_PI, 'config')
COMMON = os.path.join(DATA, 'v11_s1_random', 'data.h5')
T0, A = 20, 0.28


def load_v12(exp, seed, epoch, dev):
    cfg = load_config(os.path.join(_PI, 'config', 'exp', exp + '.yml'))
    mcfg = load_config(os.path.join(_PI, 'config', 'model', cfg.model.name, cfg.model.config + '.yml'))
    m = getattr(model_v12, cfg.model.name)(mcfg).to(dev)
    m.load_state_dict(torch.load(os.path.join(RES, exp, str(seed), 'model', f'{epoch:05d}.pth'), map_location='cpu')['model'])
    m.eval()
    pm = cfg.get('p_mask_vision', None)   # stage-2 (exp3) configs have none: vision is fully masked for t > 0
    return m, (None if pm is None else float(pm))


def batch_x(g, b0, B, dev):
    x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + B]))).to(dev)
    m = torch.from_numpy(g['self_motion'][b0:b0 + B]).float().to(dev)
    return x, m


@torch.no_grad()
def run(model, p_mask, dev, batch=50):
    g = h5py.File(COMMON, 'r')['test']
    N = g['self_vision'].shape[0]
    util.seed_all(0)
    H, E, P, HID = [], [], [], []
    for b0 in range(0, N, batch):
        x, m = batch_x(g, b0, batch, dev)
        model.init_state(x.shape[0])
        h, e, p, hid = [], [], [], []
        for t in range(100):
            pm = 0 if t == 0 else p_mask
            pr = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm)['self_vision']
            hid.append(model.last_hidden.cpu().numpy())
            h.append(model.superposition_module.state['self'][0].cpu().numpy())
            p01 = ((pr + 1) / 2).clamp(0, 1)
            e.append((p01 - (x[:, t + 1] + 1) / 2).abs().mean((1, 2, 3)).cpu().numpy())
            p.append(p01.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16))
        H.append(np.stack(h, 1)); E.append(np.stack(e, 1)); P.append(np.stack(p, 1)); HID.append(np.stack(hid, 1))
    return {'h': np.concatenate(H), 'err': np.concatenate(E), 'pred': np.concatenate(P), 'hidden': np.concatenate(HID),
            'pos': g['self_position'][()], 'layout': g['layout'][()], 'src': g['a1_source'][()]}


def tracking(d):
    N = d['h'].shape[0]
    X = d['h'].reshape(-1, 128)
    grp = np.repeat(np.arange(N)[:, None], 100, 1).reshape(-1)
    hid = d['hidden'].reshape(-1)
    # step within the hidden interval (1, 2, ...), 0 when visible
    k = np.zeros_like(d['hidden'], int)
    for t in range(100):
        k[:, t] = np.where(d['hidden'][:, t], (k[:, t - 1] if t else 0) + 1, 0)
    k = k.reshape(-1)
    out = {}
    for nm, Y in (('p_next', d['pos'][:, 1:101].reshape(-1, 2)), ('p_now_v4conv', d['pos'][:, :100].reshape(-1, 2))):
        Yh = np.zeros_like(Y)
        for tr, te in GroupKFold(5).split(X, Y, grp):
            Yh[te] = Ridge(alpha=1.0).fit(X[tr], Y[tr]).predict(X[te])

        def r2(m):  # noqa
            return float(1 - ((Y[m] - Yh[m]) ** 2).sum() / ((Y[m] - Y[m].mean(0)) ** 2).sum())
        out[nm] = {'all': r2(np.ones_like(hid)), 'visible': r2(~hid), 'hidden': r2(hid),
                   'hidden_step1_5': r2((k >= 1) & (k <= 5)), 'hidden_step6_10': r2((k >= 6) & (k <= 10)),
                   'hidden_step11_15': r2((k >= 11) & (k <= 15)),
                   'err_median_visible': float(np.median(np.linalg.norm(Y[~hid] - Yh[~hid], axis=1))),
                   'err_median_hidden': float(np.median(np.linalg.norm(Y[hid] - Yh[hid], axis=1)))}
    out['hidden_frac'] = float(hid.reshape(N, 100)[:, 1:].mean())
    return out


def summ(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def m1_all(d, cf, cfv):
    out, per = {}, {}
    for nm, fm in (('all', None), ('hidden', d['hidden']), ('visible', ~d['hidden'])):
        delta, cp = m1c(d, d['src'], fm)
        S, cd = m1d(d, d['src'], cf, cfv, fm)
        col = (S['Red'] - S['Green'])[d['layout'] == 1]    # colour criterion (v11 Sec.7.13): swap episodes
        out[nm] = {'M1c_c_pred': summ(cp), 'M1c_Delta': {X: float(np.nanmean(v)) for X, v in delta.items()},
                   'M1d_c_cf': summ(cd), 'M1d_S': {X: float(np.nanmean(v)) for X, v in S.items()},
                   'colour_swap_SRed_minus_SGreen': summ(col)}
        per[f'{nm}_c_pred'] = cp; per[f'{nm}_c_cf'] = cd
    return out, per


@torch.no_grad()
def freeze(model, p_mask, dev, K, hold, batch=50):
    g = h5py.File(COMMON, 'r')['test']
    N = g['self_vision'].shape[0]
    out = []
    for b0 in range(0, N, batch):
        x, m = batch_x(g, b0, batch, dev)
        B = x.shape[0]
        util.seed_all(0)
        model.init_state(B)
        for t in range(T0):
            pm = 0 if t == 0 else p_mask
            model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm)
        P = model.compute_pleasure(x[:, T0]) if hold else torch.zeros(B, 8, device=dev)
        hid = torch.ones(B, dtype=torch.bool, device=dev)
        preds = []
        for k in range(K):
            pr = model({'self_vision': x[:, T0 + k], 'self_motion': m[:, T0 + k]}, 1.0, 1.0, p1_override=P,
                       hidden_override=hid)['self_vision']
            preds.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
        out.append(np.stack(preds, 1))
    return np.concatenate(out)


@torch.no_grad()
def alignment(model):
    g = h5py.File(COMMON, 'r')['test']
    keep = np.where(g['layout'][()] == 0)[0]
    ts = np.arange(0, 100, 5)
    sv = g['self_vision'][()][keep][:, ts].reshape(-1, 16, 64, 3)
    ov = g['other_vision'][()][keep][:, ts].reshape(-1, 16, 64, 3)
    dev = next(model.parameters()).device

    def enc(mod, fr):  # noqa
        o = []
        for i in range(0, len(fr), 2000):
            o.append(mod(torch.from_numpy(util.scale_vision(util.transpose_vision(fr[i:i + 2000], dim=4))).to(dev)).cpu().numpy())
        return np.concatenate(o)
    ov_a1 = enc(model.other_vision_encoder_module, sv)
    return {'cos_ov(A1)_sv(A2)': cosm(ov_a1, enc(model.self_vision_encoder_module, ov)),
            'cos_ov(A1)_sv(A1)': cosm(ov_a1, enc(model.self_vision_encoder_module, sv))}


@torch.no_grad()
def p2test(model, p_mask, dev, flag, K=10, batch=50):
    g = h5py.File(COMMON, 'r')['test']
    sv_all, sp_all, op_all = g['self_vision'], g['self_position'][()], g['other_position'][()]
    alpha, beta = calibrate(sv_all[:, :100:5], sp_all[:, :100:5], op_all[:, :100:5])
    N = sp_all.shape[0]
    est = {}
    for b0 in range(0, N, batch):
        x, m = batch_x(g, b0, batch, dev)
        B = x.shape[0]
        sp = sp_all[b0:b0 + batch, T0:T0 + K]
        keys = [('ctrl', None)] + [(f'{sg}{k}', (sgn, k)) for sg, sgn in (('p', 1.0), ('m', -1.0)) for k in range(8)]
        for key, spec in keys:
            # process-2 = [motion 0, pleasure, flag]; the control has the same flag (only pleasure differs)
            Q = torch.zeros(B, 11, device=dev)
            Q[:, 10] = float(flag)
            if spec is not None:
                Q[:, 2 + spec[1]] = spec[0] * A
            util.seed_all(0)
            model.init_state(B)
            prs = []
            for t in range(T0 + K):
                pm = 0 if t == 0 else p_mask
                pr = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm,
                           p2_override=Q if t >= T0 else None)['self_vision']
                if t >= T0:
                    prs.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
            e = np.stack([read_a2(prs[i], sp[:, i], alpha, beta) for i in range(K)], 1)
            est.setdefault(key, []).append(e)
    est = {k: np.concatenate(v) for k, v in est.items()}
    r, per = {}, {}
    for sg in ('p', 'm'):
        D = np.stack([est[f'{sg}{k}'] - est['ctrl'] for k in range(8)], 1)
        proj = np.einsum('nksc,jc->nkjs', D, BIN_UNIT)
        own = np.stack([proj[:, k, k] for k in range(8)], 1)
        other = np.stack([np.nanmean(np.delete(proj[:, :, k], k, axis=1), 1) for k in range(8)], 1)
        spec_ = np.nanmean(own[:, :, -1] - other[:, :, -1], 1)
        r[sg] = {'d_spec': summ(spec_), 'own_proj_step10_bins0167': float(np.nanmean(own[:, [0, 1, 6, 7], -1])),
                 'own_proj_step10_bins2to5': float(np.nanmean(own[:, 2:6, -1])),
                 'own_proj_step10_per_bin': np.nanmean(own[:, :, -1], 0).round(4).tolist()}
        per[sg] = spec_
    r['frac_read'] = float(np.mean(~np.isnan(est['ctrl'][..., 0])))
    # Sec.10 item 4: does the shift reverse between pleasure (+a) and displeasure (-a)? per episode
    r['p_minus_m'] = summ(per['p'] - per['m'])
    return r, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, nargs='+', default=[60, 100, 150])
    ap.add_argument('--p2_epoch', type=int, default=150)
    ap.add_argument('--tag', default='')
    ap.add_argument('--exp_fmt', default='v12_trial_{}_mse', help='e.g. v12_trial_pb_{}_mse for (P-b)')
    ap.add_argument('--prefix', default='', help='output file prefix, e.g. pb_')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='output dir under data/result/v12 (default: trial)')
    ap.add_argument('--no_p2', action='store_true', help='skip the process-2 test (G5b is computed by eval_g5_v12.py)')
    ap.add_argument('--conds', nargs='+', default=['random', 'mixed'], help='E2 (Sec.53) has the experienced run only')
    args = ap.parse_args()
    out_dir = OUT if args.out is None else os.path.join(RES, 'v12', args.out)
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    fcf = h5py.File(os.path.join(RES, 'v11', 'cf', 'cf_views.h5'), 'r')
    cf, cfv = fcf['cf_vision'][()], fcf['valid'][()]
    fv = h5py.File(os.path.join(RES, 'v11', 'cf', 'cf_dist_views.h5'), 'r')
    views, vvalid = fv['views'][()], fv['valid'][()]
    for e in args.epochs:
        per_all = {}
        for c in args.conds:
            model, p_mask = load_v12(args.exp_fmt.format(c), args.seed, e, dev)
            d = run(model, p_mask, dev)
            r = {'exp': args.exp_fmt.format(c), 'epoch': e, 'err_mean_visible': float(d['err'][~d['hidden']].mean()),
                 'err_mean_hidden': float(d['err'][d['hidden']].mean())}
            r['tracking'] = tracking(d)
            r['M1'], per = m1_all(d, cf, cfv)
            del d
            r['freeze'] = {}
            for K in (15, 20):
                for variant, hold in (('hold', True), ('zero', False)):
                    pr = freeze(model, p_mask, dev, K, hold)
                    ds = dstar(pr, views, vvalid)
                    s = slopes(ds, K)
                    gg = g_of(s)
                    r['freeze'][f'K{K}_{variant}'] = {'g': summ(gg), 'slope': {X: float(np.nanmean(s[:, i])) for i, X in enumerate(('Red', 'Green', 'Blue', 'Cyan'))},
                                                     'dstar_curve_red': np.nanmean(ds[:, 0], 0).round(3).tolist(),
                                                     'dstar_curves_RGBC': np.nanmean(ds, 0).round(3).tolist()}
                    per[f'freeze_K{K}_{variant}'] = gg
            r['alignment'] = alignment(model)
            if e == args.p2_epoch and not args.no_p2:
                r['p2test'] = {}
                for flag in (0, 1):
                    r['p2test'][f'flag{flag}'], pp = p2test(model, p_mask, dev, flag)
                    per[f'p2_flag{flag}_p'] = pp['p']
                    per[f'p2_flag{flag}_p_minus_m'] = pp['p'] - pp['m']
            per_all[c] = per
            with open(os.path.join(out_dir, f'{args.prefix}eval_{c}_ep{e}{args.tag}.json'), 'w') as f:
                json.dump(r, f, indent=1)
            print(c, e, json.dumps({'track': r['tracking']['p_next'], 'M1_hidden': {k: r['M1']['hidden'][k] for k in ('M1c_c_pred', 'M1d_c_cf')},
                                    'freeze_K15_hold': r['freeze']['K15_hold']['g'], 'align': r['alignment']}), flush=True)
        if len(per_all) < 2:
            continue
        paired = {k: summ(per_all['mixed'][k] - per_all['random'][k]) for k in per_all['mixed']}
        np.savez_compressed(os.path.join(out_dir, f'{args.prefix}per_ep{e}{args.tag}.npz'),
                            **{f'{c}__{k}': v for c in per_all for k, v in per_all[c].items()})
        with open(os.path.join(out_dir, f'{args.prefix}paired_ep{e}{args.tag}.json'), 'w') as f:
            json.dump(paired, f, indent=1)
        print('paired', e, json.dumps({k: round(v['mean'], 4) for k, v in paired.items()}), flush=True)


if __name__ == '__main__':
    main()
