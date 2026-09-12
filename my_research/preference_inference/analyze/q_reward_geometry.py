"""
Reward-geometry check (2026-09-09, user hypothesis during v5 Fig.4c review).

Hypothesis: both reward landmarks sit on the x=-9 side (Red -9,+9 / Green
-9,-9), so A-1's probe-Q field varies mostly along one axis. If probe-Q
carries one position axis but not the other, the downstream place map can
only organise along that axis -- i.e. v5's one-axis h2 map would be an
ENVIRONMENT constraint, not a generic "value input -> 1D map" law.

probe-Q comes from the FROZEN A-1 critic (config's probe_critic), so this
is model-independent -- run once.

Uses natural-trajectory data (self_vision + self_position per episode/t).

Tests:
  1. Ridge-regress the 8-dim probe-Q -> x and -> y separately; compare R^2.
  2. Bin positions to a coarse grid, mean scalar value V per bin, finite-diff
     |dV/dx| vs |dV/dy|.
  3. Save V heatmap + argmax-probe quiver.

Usage (from /work/my_research/preference_inference, NO CUDA_VISIBLE_DEVICES clamp):
  DEVICE=cuda:6 python analyze/q_reward_geometry.py --exp_config r3_a_direct_1000pretrain_400 --epoch 400 --data r3_stay
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

_PI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('/work', _PI):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as models  # noqa
import util  # noqa

DEV = torch.device(os.environ.get('DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
OUT = 'data/result/baseline_v4'


class A:
    def __init__(s, e, se):
        s.exp_config, s.seed = e, se


def ridge_r2(X, y, lam=1.0):
    Xc = X - X.mean(0)
    yc = y - y.mean()
    beta = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ yc)
    pred = Xc @ beta
    return 1 - ((yc - pred) ** 2).sum() / (yc ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', default='r3_a_direct_1000pretrain_400')
    ap.add_argument('--epoch', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data', default='r3_stay')
    ap.add_argument('--n_ep', type=int, default=1500)
    ap.add_argument('--T', type=int, default=100)
    ap.add_argument('--label', default=None)
    args = ap.parse_args()
    label = args.label or f'{args.exp_config}_{args.data}'

    ec = util.gen_exp_config(A(args.exp_config, args.seed))
    mc = util.gen_model_config(ec)
    _, mdir, _ = util.gen_dirs(A(args.exp_config, args.seed), test=False)
    net = getattr(models, ec.model.name)(mc).to(DEV)
    util.load_model(mdir, args.epoch, net)
    net.eval()
    K = net.probe_actions.size(0)
    probe_dirs = net.probe_actions.cpu().numpy()

    with h5py.File(f'data/data/{args.data}/data.h5', 'r') as f:
        n = min(args.n_ep, f['train/self_vision'].shape[0])
        sv = f['train/self_vision'][:n, :args.T]
        sp = f['train/self_position'][:n, :args.T]
    sv = sv.reshape(-1, *sv.shape[2:]).astype(np.float32)
    if sv.max() > 1.5:
        sv = sv / 255.0
    pos = sp.reshape(-1, 2).astype(np.float32)
    print(f'{len(sv)} frames, K={K}, model={args.exp_config} ep{args.epoch}, data={args.data}')

    Q = np.zeros((len(sv), K), dtype=np.float32)
    B = 256
    with torch.no_grad():
        for b0 in range(0, len(sv), B):
            t = torch.tensor(util.scale_vision(sv[b0:b0 + B])).permute(0, 3, 1, 2).float().to(DEV)
            Q[b0:b0 + B] = net.compute_probe_q((t + 1) / 2).cpu().numpy()

    # --- test 1 ---
    r2x = ridge_r2(Q, pos[:, 0])
    r2y = ridge_r2(Q, pos[:, 1])
    r2xy = 1 - (((pos - pos.mean(0)) - (Q - Q.mean(0)) @ np.linalg.solve(
        (Q - Q.mean(0)).T @ (Q - Q.mean(0)) + np.eye(K),
        (Q - Q.mean(0)).T @ (pos - pos.mean(0)))) ** 2).sum() / ((pos - pos.mean(0)) ** 2).sum()
    print(f'\n[1] probe-Q(8) -> position, Ridge R^2:  x={r2x:.4f}  y={r2y:.4f}  '
          f'(gap x-y={r2x - r2y:+.4f})   combined={r2xy:.4f}')

    # --- test 2: coarse-grid V gradient ---
    nb = 12
    ed = np.linspace(-10, 10, nb + 1)
    cx = (ed[:-1] + ed[1:]) / 2
    Vmean = np.full((nb, nb), np.nan)
    Vmax = np.full((nb, nb), np.nan)
    ix = np.clip(np.digitize(pos[:, 0], ed) - 1, 0, nb - 1)
    iy = np.clip(np.digitize(pos[:, 1], ed) - 1, 0, nb - 1)
    qm = Q.mean(1)
    qx = Q.max(1)
    for a in range(nb):
        for b in range(nb):
            m = (ix == a) & (iy == b)
            if m.sum() >= 5:
                Vmean[a, b] = qm[m].mean()
                Vmax[a, b] = qx[m].mean()
    d = cx[1] - cx[0]
    for nm, V in [('mean_k Q', Vmean), ('max_k Q', Vmax)]:
        gx = np.gradient(np.nan_to_num(V, nan=np.nanmean(V)), axis=0) / d
        gy = np.gradient(np.nan_to_num(V, nan=np.nanmean(V)), axis=1) / d
        mask = ~np.isnan(V)
        print(f'[2] {nm}: mean|dV/dx|={np.abs(gx)[mask].mean():.4f}  '
              f'mean|dV/dy|={np.abs(gy)[mask].mean():.4f}  '
              f'ratio x/y={np.abs(gx)[mask].mean() / (np.abs(gy)[mask].mean() + 1e-9):.2f}')

    # --- plots ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, V, tt in [(ax[0], Vmean, 'mean_k probe-Q'), (ax[1], Vmax, 'max_k probe-Q')]:
        im = a.imshow(V.T, origin='lower', extent=[-10, 10, -10, 10], cmap='coolwarm')
        plt.colorbar(im, ax=a)
        a.plot(-9, 9, 'k*', ms=14); a.annotate(' Red', (-9, 9), fontsize=9)
        a.plot(-9, -9, 'k*', ms=14); a.annotate(' Green', (-9, -9), fontsize=9)
        a.set_title(tt); a.set_xlabel('x'); a.set_ylabel('y'); a.set_aspect('equal')
    amax = Q.argmax(1)
    ax[2].quiver(pos[::37, 0], pos[::37, 1], probe_dirs[amax][::37, 0], probe_dirs[amax][::37, 1],
                 scale=25, width=0.004, alpha=0.6)
    ax[2].plot(-9, 9, 'r*', ms=14); ax[2].plot(-9, -9, 'g*', ms=14)
    ax[2].set_title('argmax-probe dir'); ax[2].set_aspect('equal')
    ax[2].set_xlim(-10, 10); ax[2].set_ylim(-10, 10)
    fig.suptitle(f'{label}  probe-Q->x R^2={r2x:.2f}  ->y R^2={r2y:.2f}')
    fig.tight_layout()
    outp = os.path.join(OUT, f'q_reward_geometry_{label}.png')
    fig.savefig(outp, dpi=110)
    print(f'\nsaved {outp}')


if __name__ == '__main__':
    main()
