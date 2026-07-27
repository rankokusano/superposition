"""
Visualize visual reconstruction from v3 B-base and B-MGVE models.

For each model, decode vision from:
  - h1-only  : integration(ss, zeros)  → A-1 process contribution
  - h2-only  : integration(zeros, os)  → A-2 process contribution
  - full      : integration(ss, os)    → complete prediction

Shows actual input alongside all three reconstructions.

Usage (from /work/my_research/preference_inference inside Docker):
    python analyze/plot_vision_recon_v3.py
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
import torch.nn.functional as F

from util import load_config
import model as models

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = '/work/my_research/preference_inference/data'
DATA_H5   = os.path.join(DATA_ROOT, 'data', 'v3_b_mgve_train', 'data.h5')
SAVE_DIR  = os.path.join(DATA_ROOT, 'result', 'v3_vision_recon')

MODELS = {
    'B-base (A2-stationary data)': {
        'cls':  'SuperpositionNetworkApproachBBaseV3',
        'cfg':  '/work/my_research/preference_inference/config/model/SuperpositionNetworkApproachBBaseV3/default.yml',
        'ckpt': os.path.join(DATA_ROOT, 'result', 'v3_exp_b_base_l1', '0', 'model', '00200.pth'),
        'data': os.path.join(DATA_ROOT, 'data', 'v3_b_base_train', 'data.h5'),
    },
}

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SAMPLE  = 3     # number of episodes to visualize
TSTEPS    = [10, 30, 60, 90]   # timesteps to show
EPOCH     = 200


def sv_to_tensor(sv_np):
    """(H, W, C) uint8-ish float → (1, C, H, W) tensor"""
    return torch.tensor(sv_np.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)


def tensor_to_img(t):
    """(1, C, H, W) tensor → (H, W, C) clipped float"""
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)


def run_episode(net, sv_ep, q1_ep):
    """
    Run one episode and return reconstructions at each timestep.
    sv_ep : (T, H, W, C)
    q1_ep : (T, 1)
    Returns dict of lists: actual, h1_only, h2_only, full  each (T, H, W, C)
    """
    T = sv_ep.shape[0]
    net.init_state(1)

    results = {k: [] for k in ['actual', 'h1_only', 'h2_only', 'full']}

    with torch.no_grad():
        for t in range(T):
            sv_t = sv_to_tensor(sv_ep[t])       # (1, C, H, W)
            q1_t = torch.tensor(q1_ep[t:t+1].astype(np.float32)).to(DEVICE)  # (1,1)

            # encode
            sv_enc = net.self_vision_encoder_module(sv_t)
            ov_enc = net.other_vision_encoder_module(sv_t)

            # motion for each process
            if hasattr(net, 'value_estimator_module'):  # B-MGVE
                om_gen  = net.motion_generator_module(ov_enc)
                q2_hat  = net.value_estimator_module(ov_enc, om_gen)
                sm_in   = q1_t
                om_in   = q2_hat
            else:  # B-base
                sm_in   = q1_t
                om_in   = torch.zeros_like(q1_t)

            # SM forward (updates internal state, returns ss, os)
            ss, os = net.superposition_module(sv_enc, sm_in, ov_enc, om_in)

            # integration & decode
            zeros_s = torch.zeros_like(ss)
            zeros_o = torch.zeros_like(os)

            ss_d = F.dropout(ss, p=0.0)  # no dropout at eval
            os_d = F.dropout(os, p=0.0)

            so_full  = net.integration_module(ss_d, os_d)
            so_h1    = net.integration_module(ss_d, zeros_o)
            so_h2    = net.integration_module(zeros_s, os_d)

            pred_full = net.vision_decoder_module(so_full)
            pred_h1   = net.vision_decoder_module(so_h1)
            pred_h2   = net.vision_decoder_module(so_h2)

            results['actual'].append(sv_ep[t])
            results['full'].append(tensor_to_img(pred_full))
            results['h1_only'].append(tensor_to_img(pred_h1))
            results['h2_only'].append(tensor_to_img(pred_h2))

    return results


def plot_episode(results, model_name, ep_idx, tsteps, save_path):
    """
    Row 0: actual input
    Row 1: full reconstruction (h1+h2)
    Row 2: h1-only (A-1 process)
    Row 3: h2-only (A-2 process)
    """
    n_t = len(tsteps)
    row_labels = ['Actual', 'Full recon\n(h¹+h²)', 'h¹ only\n(A-1 process)', 'h² only\n(A-2 process)']
    keys       = ['actual', 'full', 'h1_only', 'h2_only']

    fig, axes = plt.subplots(4, n_t, figsize=(4 * n_t, 10))
    fig.suptitle(f'{model_name} – Visual Reconstruction  (episode {ep_idx}, epoch {EPOCH})', fontsize=13)

    for col, t in enumerate(tsteps):
        for row, (label, key) in enumerate(zip(row_labels, keys)):
            ax = axes[row, col]
            img = results[key][t]
            ax.imshow(img, interpolation='nearest')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=80, va='center')
            mse = None
            if key != 'actual':
                mse = np.mean((img - results['actual'][t]) ** 2)
                ax.set_title(f't={t}  MSE={mse:.4f}', fontsize=8)
            else:
                ax.set_title(f't={t}', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    for model_name, m in MODELS.items():
        print(f'\n=== {model_name} ===')
        cfg = load_config(m['cfg'])
        net = getattr(models, m['cls'])(cfg)
        ckpt = torch.load(m['ckpt'], map_location=DEVICE)
        net.load_state_dict(ckpt['model'])
        net.to(DEVICE)
        net.eval()

        print(f'Loading test data from {m["data"]}...')
        with h5py.File(m['data'], 'r') as f:
            sv_all = f['test/self_vision'][()]
            q1_all = f['test/a1_q_values'][()]
        N = sv_all.shape[0]
        print(f'  N={N}')
        sample_eps = np.linspace(0, N - 1, N_SAMPLE, dtype=int)

        for ep_idx in sample_eps:
            print(f'  Episode {ep_idx}...')
            sv_ep = sv_all[ep_idx]   # (T, H, W, C)
            q1_ep = q1_all[ep_idx]   # (T, 1)

            results = run_episode(net, sv_ep, q1_ep)

            fname = f'{model_name.replace("-","_").lower()}_ep{ep_idx:04d}.png'
            plot_episode(results, model_name, ep_idx, TSTEPS,
                         os.path.join(SAVE_DIR, fname))


if __name__ == '__main__':
    main()
