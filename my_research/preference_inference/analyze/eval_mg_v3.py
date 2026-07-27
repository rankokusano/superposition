"""
Evaluate Motion Generator (MG) accuracy for v3 B-MGVE model.

MG takes ov_enc (A-1's visual encoding of the scene) and predicts
om_generated (2D predicted motion of A-2).

Compares om_generated vs other_motion (true A-2 action from data).

Usage (from /work/my_research/preference_inference inside Docker):
    python analyze/eval_mg_v3.py
"""

import os, sys
sys.path.insert(0, '/work/my_research/preference_inference')
sys.path.append('/work')
sys.path.append('/work/simulation')

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

DATA_ROOT  = '/work/my_research/preference_inference/data'
MODEL_PATH = os.path.join(DATA_ROOT, 'result', 'v3_exp_b_mgve', '0', 'model', '00200.pth')
DATA_H5    = os.path.join(DATA_ROOT, 'data', 'v3_b_mgve_train', 'data.h5')
MODEL_CFG  = '/work/my_research/preference_inference/config/model/SuperpositionNetworkApproachBMGVEV3/default.yml'
SAVE_DIR   = os.path.join(DATA_ROOT, 'result', 'v3_mg_eval')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── load model ───────────────────────────────────────────────────────────
    from util import load_config
    import model as models

    model_cfg = load_config(MODEL_CFG)
    net = models.SuperpositionNetworkApproachBMGVEV3(model_cfg)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(ckpt['model'])
    net.to(DEVICE)
    net.eval()

    # ── load test data ───────────────────────────────────────────────────────
    print('Loading test data...')
    with h5py.File(DATA_H5, 'r') as f:
        sv_all  = f['test/self_vision'][()]     # (N, T, H, W, C)
        om_all  = f['test/other_motion'][()]    # (N, T, 2)  true A-2 action
        q1_all  = f['test/a1_q_values'][()]     # (N, T, 1)

    N, T, H, W, C = sv_all.shape
    print(f'  N={N}, T={T}')

    # ── run MG over all test episodes ────────────────────────────────────────
    print('Running MG forward pass...')
    om_preds = np.zeros((N, T, 2), dtype=np.float32)

    batch_size = 10
    with torch.no_grad():
        for n in range(0, N, batch_size):
            sv_b = torch.tensor(
                sv_all[n:n+batch_size].transpose(0, 1, 4, 2, 3).astype(np.float32)
            ).to(DEVICE)  # (B, T, C, H, W)
            q1_b = torch.tensor(q1_all[n:n+batch_size].astype(np.float32)).to(DEVICE)

            B = sv_b.shape[0]
            net.init_state(B)
            for t in range(T):
                sv_t = sv_b[:, t]      # (B, C, H, W)
                ov_enc = net.other_vision_encoder_module(sv_t)          # (B, D)
                om_gen = net.motion_generator_module(ov_enc)            # (B, 2)
                om_preds[n:n+B, t] = om_gen.cpu().numpy()

            if (n // batch_size + 1) % 10 == 0:
                print(f'  {n+B}/{N} done')

    print('Done.')

    # ── metrics ──────────────────────────────────────────────────────────────
    om_true = om_all  # (N, T, 2)

    # flatten
    pred_flat = om_preds.reshape(-1, 2)
    true_flat = om_true.reshape(-1, 2)

    r2_x = r2_score(true_flat[:, 0], pred_flat[:, 0])
    r2_y = r2_score(true_flat[:, 1], pred_flat[:, 1])
    r2_xy = r2_score(true_flat, pred_flat)
    mse_x = np.mean((pred_flat[:, 0] - true_flat[:, 0])**2)
    mse_y = np.mean((pred_flat[:, 1] - true_flat[:, 1])**2)

    summary = (
        f'=== Motion Generator Evaluation (v3 B-MGVE) ===\n'
        f'N={N}, T={T}, total samples={N*T}\n'
        f'\nTrue motion range:  x=[{true_flat[:,0].min():.3f}, {true_flat[:,0].max():.3f}]'
        f'  y=[{true_flat[:,1].min():.3f}, {true_flat[:,1].max():.3f}]\n'
        f'Pred motion range:  x=[{pred_flat[:,0].min():.3f}, {pred_flat[:,0].max():.3f}]'
        f'  y=[{pred_flat[:,1].min():.3f}, {pred_flat[:,1].max():.3f}]\n'
        f'\nR² (x-axis): {r2_x:.4f}\n'
        f'R² (y-axis): {r2_y:.4f}\n'
        f'R² (joint):  {r2_xy:.4f}\n'
        f'MSE (x):     {mse_x:.6f}\n'
        f'MSE (y):     {mse_y:.6f}\n'
    )
    print(f'\n{summary}')
    with open(os.path.join(SAVE_DIR, 'mg_metrics.txt'), 'w') as f:
        f.write(summary)

    # ── scatter plot: pred vs true ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for dim, (ax, label) in enumerate(zip(axes, ['x', 'y'])):
        idx = np.random.choice(len(pred_flat), size=5000, replace=False)
        ax.scatter(true_flat[idx, dim], pred_flat[idx, dim],
                   alpha=0.2, s=2, rasterized=True)
        lim = [true_flat[:, dim].min(), true_flat[:, dim].max()]
        ax.plot(lim, lim, 'r--', linewidth=1)
        r2 = r2_x if dim == 0 else r2_y
        ax.set_xlabel(f'True A-2 motion ({label})')
        ax.set_ylabel(f'MG predicted ({label})')
        ax.set_title(f'MG accuracy ({label})  R²={r2:.3f}')
    plt.tight_layout()
    p1 = os.path.join(SAVE_DIR, 'mg_scatter.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f'Saved: {p1}')

    # ── time series: sample episodes ──────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle('MG: predicted vs true A-2 motion (sample episodes)')
    sample_eps = np.linspace(0, N - 1, 6, dtype=int)
    for idx, ep in enumerate(sample_eps):
        ax = axes[idx // 2, idx % 2]
        t_ax = np.arange(T)
        ax.plot(t_ax, om_true[ep, :, 0], 'b-',  label='true x',  alpha=0.7, linewidth=1)
        ax.plot(t_ax, om_preds[ep, :, 0], 'b--', label='pred x',  alpha=0.7, linewidth=1)
        ax.plot(t_ax, om_true[ep, :, 1], 'r-',  label='true y',  alpha=0.7, linewidth=1)
        ax.plot(t_ax, om_preds[ep, :, 1], 'r--', label='pred y',  alpha=0.7, linewidth=1)
        ax.set_title(f'Episode {ep}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Motion')
        if idx == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    p2 = os.path.join(SAVE_DIR, 'mg_timeseries.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f'Saved: {p2}')

    # ── distribution comparison ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for dim, (ax, label) in enumerate(zip(axes, ['x', 'y'])):
        ax.hist(true_flat[:, dim], bins=50, alpha=0.5, label='true', density=True)
        ax.hist(pred_flat[:, dim], bins=50, alpha=0.5, label='pred', density=True)
        ax.set_xlabel(f'Motion {label}')
        ax.set_title(f'Distribution ({label})')
        ax.legend()
    plt.tight_layout()
    p3 = os.path.join(SAVE_DIR, 'mg_distribution.png')
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f'Saved: {p3}')


if __name__ == '__main__':
    main()
