"""
v11 revised M1 metrics (docs/v11_experiment_log.md Sec.7.6) for one or more
checkpoints of the random/mixed models. Reads only; writes under
data/result/v11/<sub>/.

  R^2       h1->self, v4 pairing (h after frame t <-> p_t): in-sample on own
            training data (first n_train episodes) and random-train ->
            common-test held-out.
  M1(a')    per colour X: b_X = (pos_hat - pos) . u_X (M1 pairing, regressor
            fit on random-train h through this model), frames within 2 of
            X's reference dropped, episode means; c = b_Red - mean(b_G,B,C).
  M1(c)     regressor-free: per awake frame t, the actual step m_t =
            p_{t+1} - p_t decides v_{t+1}; "toward X" = angle(m_t, ref_X -
            p_t) <= 45 deg (frames within 2 of ref_X dropped for X);
            Delta_X = mean err(toward X) - mean err(otherwise) per episode;
            c_pred = Delta_Red - mean(Delta_G,B,C).
  M1(b')    from each common-test episode: real inputs t=0..19, then 10
            steps with vision masked (p=1) and P overridden: zeros (control),
            +a or -a in bin k only (a = 0.28, Sec.7.2); d_k(s) = (est - est_ctrl)
            . unit(theta_k); main = d_k(10)(+a) - d_k(10)(-a), mean over 8 bins.
  M2        swap episodes: b_newRed - b_oldPlace(= Green's reference now)
            (proposal), plus the registered angle criterion on cell means.
  M4        5x5 cell map of c and of the mean residual vector, per layout.
Paired differences (mixed - random) use the same test episodes.
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
from sklearn.linear_model import Ridge

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import (DATA, LAYNAME, RES, boot_ci, load_model, r2,  # noqa
                            run)
import util  # noqa
from sim_v10 import layout as L  # noqa
from v11core.pleasure import BIN_ANGLES, BIN_UNIT  # noqa

COLORS = ('Red', 'Green', 'Blue', 'Cyan')
A_MANIP = 0.28
T0, NSTEP = 20, 10
EDGES = np.linspace(-10, 10, 6)


def refs(layout_codes, color):
    return np.stack([L.LANDMARKS[LAYNAME[int(l)]][color] for l in layout_codes])[:, None, :]


def m1a_prime(reg, d):
    h = d['h'].reshape(-1, 128)
    pos = d['pos'][:, 1:101]
    est = reg.predict(h).reshape(pos.shape)
    e = est - pos
    b, valid = {}, {}
    for X in COLORS:
        u = refs(d['layout'], X) - pos
        dist = np.linalg.norm(u, axis=-1)
        b[X] = (e * u / np.maximum(dist[..., None], 1e-9)).sum(-1)
        valid[X] = dist >= 2
    ep = {X: np.array([b[X][i][valid[X][i]].mean() for i in range(len(pos))]) for X in COLORS}
    c = ep['Red'] - (ep['Green'] + ep['Blue'] + ep['Cyan']) / 3
    return ep, c, b, valid, e, pos


def m1c(d, src):
    pos = d['pos']
    m = pos[:, 1:101] - pos[:, :100]
    p = pos[:, :100]
    awake = (src[:, :100] != 3) & (np.linalg.norm(m, axis=-1) > 1e-6)
    err = d['err']
    delta = {}
    for X in COLORS:
        u = refs(d['layout'], X) - p
        dist = np.linalg.norm(u, axis=-1)
        cosang = (m * u).sum(-1) / np.maximum(np.linalg.norm(m, axis=-1) * dist, 1e-9)
        toward = cosang >= np.cos(np.deg2rad(45))
        ok = awake & (dist >= 2)
        dl = np.full(len(pos), np.nan)
        for i in range(len(pos)):
            a_, b_ = ok[i] & toward[i], ok[i] & ~toward[i]
            if a_.any() and b_.any():
                dl[i] = err[i][a_].mean() - err[i][b_].mean()
        delta[X] = dl
    c = delta['Red'] - (delta['Green'] + delta['Blue'] + delta['Cyan']) / 3
    return delta, c


@torch.no_grad()
def m1b_prime(model, p_mask, reg, h5path, device, batch=50):
    g = h5py.File(h5path, 'r')['test']
    n = g['layout'].shape[0]
    W = torch.tensor(reg.coef_.T, dtype=torch.float32, device=device)  # (128, 2)
    out = {'plus': [], 'minus': []}
    for b0 in range(0, n, batch):
        sv = g['self_vision'][b0:b0 + batch]
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(sv))).to(device)
        B = x.shape[0]
        util.seed_all(0)
        model.init_state(B)
        for t in range(T0):
            pm = 0 if t == 0 else p_mask
            model({'self_vision': x[:, t]}, pm, pm)
        snap = copy.deepcopy(model.superposition_module.state)

        def branch(P):
            model.superposition_module.state = copy.deepcopy(snap)
            util.seed_all(1)
            ests = []
            for s in range(NSTEP):
                model({'self_vision': x[:, T0 + s]}, 1.0, 1.0, p1_override=P)
                ests.append(model.superposition_module.state['self'][0] @ W)
            return torch.stack(ests, 1)  # (B, NSTEP, 2); intercept cancels in differences

        ctrl = branch(torch.zeros(B, 8, device=device))
        for sign, key in ((1.0, 'plus'), (-1.0, 'minus')):
            dk = []
            for k in range(8):
                P = torch.zeros(B, 8, device=device)
                P[:, k] = sign * A_MANIP
                diff = branch(P) - ctrl
                uk = torch.tensor(BIN_UNIT[k], dtype=torch.float32, device=device)
                dk.append((diff @ uk).cpu().numpy())  # (B, NSTEP)
            out[key].append(np.stack(dk, 1))  # (B, 8, NSTEP)
    return {k: np.concatenate(v) for k, v in out.items()}  # (N, 8, NSTEP)


def summarize_ci(v):
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', required=True, help='e.g. v11_trial_{}_mse')
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_train', type=int, default=1000)
    ap.add_argument('--sub', default='trial_m1')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    out = os.path.join(RES, 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rnd = os.path.join(DATA, 'v11_s1_random', 'data.h5')
    src_test = h5py.File(rnd, 'r')['test']['a1_source'][()]
    S, per_ep, maps = {}, {}, {}
    for cond in ('random', 'mixed'):
        exp = args.exp_fmt.format(cond)
        model, p_mask = load_model(exp, args.seed, args.epoch, dev)
        own = os.path.join(DATA, f'v11_s1_{cond}', 'data.h5')
        d_tr = run(model, p_mask, rnd, 'train', args.n_train, dev)
        d_te = run(model, p_mask, rnd, 'test', 0, dev)
        d_own = d_tr if cond == 'random' else run(model, p_mask, own, 'train', args.n_train, dev)
        s = {'exp': exp, 'epoch': args.epoch, 'seed': args.seed}
        h_own = d_own['h'].reshape(-1, 128)
        p_own = d_own['pos'][:, :100].reshape(-1, 2)
        s['r2_v4conv_insample_own_train'] = r2(p_own, Ridge(alpha=1.0).fit(h_own, p_own).predict(h_own))
        reg_v4 = Ridge(alpha=1.0).fit(d_tr['h'].reshape(-1, 128), d_tr['pos'][:, :100].reshape(-1, 2))
        s['r2_v4conv_random_train_to_common_test'] = r2(d_te['pos'][:, :100].reshape(-1, 2),
                                                        reg_v4.predict(d_te['h'].reshape(-1, 128)))
        reg = Ridge(alpha=1.0).fit(d_tr['h'].reshape(-1, 128), d_tr['pos'][:, 1:101].reshape(-1, 2))

        ep_b, c, b, valid, e, pos = m1a_prime(reg, d_te)
        s['M1a_prime'] = {'c_red_specific': summarize_ci(c),
                          'b': {X: summarize_ci(v) for X, v in ep_b.items()},
                          'red_minus_green': summarize_ci(ep_b['Red'] - ep_b['Green']),
                          'per_layout_c': {LAYNAME[k]: summarize_ci(c[d_te['layout'] == k]) for k in (0, 1)}}
        delta, cp = m1c(d_te, src_test)
        s['M1c'] = {'c_pred': summarize_ci(cp), 'Delta': {X: summarize_ci(v) for X, v in delta.items()},
                    'per_layout_c_pred': {LAYNAME[k]: summarize_ci(cp[d_te['layout'] == k]) for k in (0, 1)}}
        mb = m1b_prime(model, p_mask, reg, rnd, dev)
        dplus10, dminus10 = mb['plus'][:, :, -1], mb['minus'][:, :, -1]
        contrast = dplus10 - dminus10  # (N, 8)
        sl = np.polyfit(np.arange(1, NSTEP + 1), (mb['plus'] - mb['minus']).mean((0, 1)), 1)[0]
        s['M1b_prime'] = {
            'main_mean8': summarize_ci(contrast.mean(1)),
            'in_dist_bins_0167': summarize_ci(contrast[:, [0, 1, 6, 7]].mean(1)),
            'other_bins_2_5': summarize_ci(contrast[:, 2:6].mean(1)),
            'per_bin_contrast': contrast.mean(0).round(4).tolist(),
            'per_bin_plus': dplus10.mean(0).round(4).tolist(),
            'per_bin_minus': dminus10.mean(0).round(4).tolist(),
            'contrast_curve_steps1_10': (mb['plus'] - mb['minus']).mean((0, 1)).round(4).tolist(),
            'contrast_slope_per_step': float(sl), 'a': A_MANIP}
        swap = d_te['layout'] == 1
        s['M2_proposal_swap_newRed_minus_oldPlace'] = summarize_ci((ep_b['Red'] - ep_b['Green'])[swap])
        # cell maps (M4) and registered M2 angle criterion
        lay_f = np.repeat(d_te['layout'][:, None], 100, 1)
        cframe = b['Red'] - (b['Green'] + b['Blue'] + b['Cyan']) / 3
        vall = valid['Red'] & valid['Green'] & valid['Blue'] & valid['Cyan']
        ix = np.clip(np.digitize(pos[..., 0], EDGES) - 1, 0, 4)
        iy = np.clip(np.digitize(pos[..., 1], EDGES) - 1, 0, 4)
        cm, em, angs = {}, {}, {}
        for k in (0, 1):
            C = np.full((5, 5), np.nan); E = np.full((5, 5, 2), np.nan)
            ang_new, ang_old = [], []
            for i in range(5):
                for j in range(5):
                    sel = (lay_f == k) & (ix == i) & (iy == j)
                    if sel.sum() < 50:
                        continue
                    E[i, j] = e[sel].mean(0)
                    if (sel & vall).sum() >= 50:
                        C[i, j] = cframe[sel & vall].mean()
                    ctr = np.array([(EDGES[i] + EDGES[i + 1]) / 2, (EDGES[j] + EDGES[j + 1]) / 2])
                    for X, lst in (('Red', ang_new), ('Green', ang_old)):
                        r = L.LANDMARKS[LAYNAME[k]][X] - ctr
                        a1 = np.arctan2(E[i, j, 1], E[i, j, 0]); a2 = np.arctan2(r[1], r[0])
                        lst.append(np.degrees(np.abs(np.angle(np.exp(1j * (a1 - a2))))))
            cm[k], em[k] = C, E
            angs[LAYNAME[k]] = {'to_red_deg': float(np.mean(ang_new)), 'to_other_place_deg': float(np.mean(ang_old))}
        s['M2_registered_angle'] = angs
        maps[cond] = (cm, em)
        S[cond] = s
        per_ep[cond] = {'c': c, 'cp': cp, 'b1': contrast.mean(1), 'm2': (ep_b['Red'] - ep_b['Green'])[swap],
                        'b_red': ep_b['Red']}
        print(cond, args.epoch, 'R2', round(s['r2_v4conv_insample_own_train'], 4),
              'c', s['M1a_prime']['c_red_specific'], 'c_pred', s['M1c']['c_pred'],
              'b1', s['M1b_prime']['main_mean8'], flush=True)

    S['paired_mixed_minus_random'] = {
        'M1a_prime_c': summarize_ci(per_ep['mixed']['c'] - per_ep['random']['c']),
        'M1c_c_pred': summarize_ci(per_ep['mixed']['cp'] - per_ep['random']['cp']),
        'M1b_prime_main': summarize_ci(per_ep['mixed']['b1'] - per_ep['random']['b1']),
        'M2_proposal': summarize_ci(per_ep['mixed']['m2'] - per_ep['random']['m2']),
        'registered_b_red': summarize_ci(per_ep['mixed']['b_red'] - per_ep['random']['b_red']),
    }
    tag = f'{args.exp_fmt.format("X")}_s{args.seed}_ep{args.epoch}'
    with open(os.path.join(out, f'm1_{tag}.json'), 'w') as f:
        json.dump(S, f, indent=1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for r_, cond in enumerate(('random', 'mixed')):
        cm, em = maps[cond]
        for k in (0, 1):
            ax = axes[r_, k]
            im = ax.imshow(cm[k].T, origin='lower', extent=[-10, 10, -10, 10], cmap='RdBu_r', vmin=-0.6, vmax=0.6)
            ctr = (EDGES[:-1] + EDGES[1:]) / 2
            X, Y = np.meshgrid(ctr, ctr, indexing='ij')
            ax.quiver(X, Y, em[k][..., 0], em[k][..., 1], angles='xy', scale_units='xy', scale=0.5)
            ax.plot(*L.LANDMARKS[LAYNAME[k]]['Red'], 'o', color='red', mec='w', ms=10)
            ax.plot(*L.LANDMARKS[LAYNAME[k]]['Green'], 'o', color='lime', mec='k', ms=10)
            ax.set_title(f'{cond} / {LAYNAME[k]}', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'M4: colour = red-specific bias c per 4x4 cell; arrows = mean residual (x2), common random test\n{tag}', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'm4_{tag}.png'), dpi=120)
    plt.close(fig)
    print(json.dumps(S['paired_mixed_minus_random']))


if __name__ == '__main__':
    main()
