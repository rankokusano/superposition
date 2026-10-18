"""
v11 reference test: pleasure input fed to process-2 (docs/v11_experiment_log.md
Sec.7.33, registered before computing). Process-1 keeps its normal inputs; from
t = 20, process-2 receives Q = +-a e_k (a = 0.28) or zero for K = 10 steps. The
A-2 position is read from each one-step prediction: dark-pixel weights
w = clip((0.2 - max_c)/0.2, 0, 1), circular column centroid -> bearing phi,
mass M -> distance via log(dist) = alpha + beta log(count) calibrated on the true
views (black = all channels < 0.1); position = true A-1 position + dist u(phi).
Main index d_spec = mean_k [D_k(10).u_k - mean_{j!=k} D_j(10).u_k], D_k = est(+a e_k) - est(0).
Writes JSON + figure under data/result/v11/main/p2test/.
"""

import argparse
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
from eval_trial_v11 import DATA, RES, boot_ci, load_model  # noqa
import util  # noqa
from v11core.pleasure import BIN_UNIT, COL_ANGLES  # noqa

T0, K, A = 20, 10, 0.28
MIN_MASS = 2.0


def calibrate(sv, sp, op):
    """True views: black-pixel count vs true A-1 -> A-2 distance (frames with >= 3 black pixels)."""
    v = sv.astype(np.float32) / 255.0
    cnt = (v.max(-1) < 0.1).sum((-1, -2)).reshape(-1)
    dist = np.linalg.norm(op - sp, axis=-1).reshape(-1)
    m = cnt >= 3
    beta, alpha = np.polyfit(np.log(cnt[m]), np.log(dist[m]), 1)
    return float(alpha), float(beta)


def read_a2(pred, sp, alpha, beta):
    """pred (B, H, W, 3) in [0,1]; sp (B, 2) -> est position (B, 2) with NaN where mass < MIN_MASS."""
    w = np.clip((0.2 - pred.max(-1)) / 0.2, 0, 1)            # (B, H, W)
    col = w.sum(1)                                             # (B, W)
    M = col.sum(1)
    z = (col * np.exp(1j * COL_ANGLES)[None]).sum(1)
    phi = np.angle(z)
    dist = np.exp(alpha + beta * np.log(np.maximum(M, 1e-6)))
    est = sp + dist[:, None] * np.stack([np.cos(phi), np.sin(phi)], 1)
    est[M < MIN_MASS] = np.nan
    return est


@torch.no_grad()
def branch_preds(model, p_mask, x, Q):
    """Run t = 0..T0+K-1; from T0 feed p2_override = Q (or None before). Returns preds (B, K, H, W, 3)."""
    B = x.shape[0]
    util.seed_all(0)
    model.init_state(B)
    out = []
    for t in range(T0 + K):
        pm = 0 if t == 0 else p_mask
        q = Q if t >= T0 else None
        pr = model({'self_vision': x[:, t]}, pm, pm, p2_override=q)['self_vision']
        if t >= T0:
            out.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
    return np.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 2])
    ap.add_argument('--exp_fmt', default='v11_s1_{}_mse')
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--sub', default='main/p2test')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    out = os.path.join(RES, 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    g = h5py.File(os.path.join(DATA, 'v11_s1_random', 'data.h5'), 'r')['test']
    sv_all, sp_all, op_all = g['self_vision'][()], g['self_position'][()], g['other_position'][()]
    alpha, beta = calibrate(sv_all[:, :100:5], sp_all[:, :100:5], op_all[:, :100:5])
    N = len(sv_all)
    res = {'calibration': {'alpha': alpha, 'beta': beta}}
    per = {}
    for s in args.seeds:
        for cond in ('random', 'mixed'):
            model, p_mask = load_model(args.exp_fmt.format(cond), s, args.epoch, dev)
            est = {}  # key -> (N, K, 2)
            for b0 in range(0, N, 50):
                x = torch.from_numpy(util.scale_vision(util.transpose_vision(sv_all[b0:b0 + 50]))).to(dev)
                B = x.shape[0]
                sp = sp_all[b0:b0 + 50, T0:T0 + K]
                keys = [('ctrl', None)] + [(f'{sg}{k}', (sgn, k)) for sg, sgn in (('p', 1.0), ('m', -1.0)) for k in range(8)]
                for key, spec in keys:
                    Q = torch.zeros(B, 8, device=dev)
                    if spec is not None:
                        Q[:, spec[1]] = spec[0] * A
                    pr = branch_preds(model, p_mask, x, Q)
                    e = np.stack([read_a2(pr[:, i], sp[:, i], alpha, beta) for i in range(K)], 1)
                    est.setdefault(key, []).append(e)
            est = {k: np.concatenate(v) for k, v in est.items()}
            r = {}
            for sg in ('p', 'm'):
                D = np.stack([est[f'{sg}{k}'] - est['ctrl'] for k in range(8)], 1)   # (N, 8, K, 2)
                proj = np.einsum('nksc,jc->nkjs', D, BIN_UNIT)                       # (N, k manip, j dir, K)
                own = np.stack([proj[:, k, k] for k in range(8)], 1)                 # (N, 8, K)
                other = np.stack([np.nanmean(np.delete(proj[:, :, k], k, axis=1), 1) for k in range(8)], 1)
                spec_ = np.nanmean(own[:, :, -1] - other[:, :, -1], 1)               # (N,)
                v = spec_[~np.isnan(spec_)]
                slope = np.polyfit(np.arange(1, K + 1), np.nanmean(own - other, (0, 1)), 1)[0]
                r[sg] = {'d_spec': {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))},
                         'own_proj_step10_per_bin': np.nanmean(own[:, :, -1], 0).round(4).tolist(),
                         'own_proj_step10_bins0167': float(np.nanmean(own[:, [0, 1, 6, 7], -1])),
                         'own_proj_step10_bins2to5': float(np.nanmean(own[:, 2:6, -1])),
                         'spec_curve_steps1_10': np.nanmean(own - other, (0, 1)).round(4).tolist(),
                         'spec_slope_per_step': float(slope),
                         'mean_vec_step10': np.nanmean(D[:, :, -1], 0).round(4).tolist()}
                per[(cond, s, sg)] = spec_
            r['frac_frames_with_A2_read'] = float(np.mean(~np.isnan(est['ctrl'][..., 0])))
            res[f'{cond}_s{s}'] = r
            print(cond, s, 'd_spec +a', r['p']['d_spec'], '-a', r['m']['d_spec']['mean'], 'read frac', r['frac_frames_with_A2_read'], flush=True)
    paired = {}
    for s in args.seeds:
        for sg in ('p', 'm'):
            dd = per[('mixed', s, sg)] - per[('random', s, sg)]
            dd = dd[~np.isnan(dd)]
            paired[f's{s}_{sg}'] = {'mean': float(dd.mean()), 'ci95': boot_ci(dd), 'n': int(len(dd))}
    res['paired_mixed_minus_random'] = paired
    ok = all(res[f'mixed_s{s}']['p']['d_spec']['mean'] > 0 and res[f'mixed_s{s}']['p']['d_spec']['ci95'][0] > 0
             and paired[f's{s}_p']['ci95'][0] > 0 for s in args.seeds)
    res['reference_positive'] = bool(ok)
    with open(os.path.join(out, f'p2test_ep{args.epoch}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    fig, axes = plt.subplots(2, 2 * len(args.seeds), figsize=(5 * 2 * len(args.seeds), 9), squeeze=False)
    for j, s in enumerate(args.seeds):
        for i, cond in enumerate(('random', 'mixed')):
            for c, sg in enumerate(('p', 'm')):
                a_ = axes[i, 2 * j + c]
                V = np.array(res[f'{cond}_s{s}'][sg]['mean_vec_step10'])
                lim = max(0.5, np.nanmax(np.abs(V)) * 1.3)
                for k in range(8):
                    u = BIN_UNIT[k] * lim * 0.95
                    a_.plot([0, u[0]], [0, u[1]], color='0.85', lw=0.8)
                    a_.text(u[0], u[1], str(k), fontsize=7, color='0.5')
                    a_.arrow(0, 0, V[k, 0], V[k, 1], color='tab:red' if sg == 'p' else 'tab:green', width=lim * 0.008,
                             length_includes_head=True)
                a_.set_xlim(-lim, lim); a_.set_ylim(-lim, lim); a_.set_aspect('equal')
                a_.set_title(f'{cond} s{s}, process-2 Q = {"+" if sg == "p" else "-"}a in bin k\n'
                             f'd_spec = {res[f"{cond}_s{s}"][sg]["d_spec"]["mean"]:+.3f}', fontsize=8)
    fig.suptitle('Reference test: shift of the A-2 position read from the predicted image after 10 steps (arena units); '
                 'grey = bin directions', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, f'p2test_ep{args.epoch}.png'), dpi=110)
    print('reference_positive', ok, json.dumps(paired))


if __name__ == '__main__':
    main()
