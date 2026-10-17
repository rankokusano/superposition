"""
v11 Fig.5 (perspective taking; instructions Sec.7.1, docs/v11_experiment_log.md
Sec.7.18 items 15-16) for one Fig.5 Autoencoder, on the original `grid` and
`v10_swap_grid` (read-only links). The Autoencoder is run directly on the
grid (no saved.h5): input = A-1's view with A-2 drawn, self_vision[o, s].

Numbers (same as analyze_vpt.py / vpt_metrics_v10.py, margin 1):
  self_self   = |self_vision_no_agent[o,s]  - self_rec[o,s]|
  other_other = |other_vision_no_agent[o,s] - other_rec[o,s]|
  self_other  = |self_vision_no_agent[o,s]  - other_rec[o,s]|
  other_self  = |other_vision_no_agent[o,s] - self_rec[o,s]|
  (pixel mean per frame, then mean/std over frames inside the margin)
Error maps (analyze_vpt.py's map): for chosen (A-1, A-2) positions,
  |true_images[x, y] - other_rec| over all 21x21 positions (true_images =
  A-1's agent-free view at each grid point), A-1/A-2 marked; also for
  self_rec. Descriptive extra: fraction of (o, s) pairs (margin 1, both
  layouts) whose other_rec error map has its minimum within 1 cell of A-2's
  true position (and self_rec within 1 cell of A-1's).
Writes JSON, PNG and .npz under data/result/v11/<sub>/.
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
from eval_trial_v11 import DATA, RES, load_model  # noqa
import util  # noqa

GRIDS = {'original': os.path.join(DATA, 'grid', 'data.h5'),
         'swap_rg': os.path.join(DATA, 'v10_swap_grid', 'data.h5')}
SIZE = 21
CONFIGS = [((-5, 5), (5, -5)), ((5, 5), (-5, -5)), ((0, -7), (0, 7)), ((-7, 0), (7, 0))]  # (A-1, A-2)


def f32(a):
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)


def idx(p):
    return (p[0] + 10) * SIZE + (p[1] + 10)


@torch.no_grad()
def rec_all(ae, frames, device, bs=441):
    out_s, out_o = [], []
    for b in range(0, len(frames), bs):
        v = f32(frames[b:b + bs])
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(v, dim=4))).to(device)
        r = ae(x, decode_from_other=True)
        out_s.append(((r['self_vision'] + 1) / 2).permute(0, 2, 3, 1).cpu().numpy())
        out_o.append(((r['other_vision'] + 1) / 2).permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out_s), np.concatenate(out_o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ae', required=True, help='Fig.5 Autoencoder exp_config')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epoch', type=int, default=100)
    ap.add_argument('--margin', type=int, default=1)
    ap.add_argument('--sub', default='fig5')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--ckpt_root', default=None, help='read-only checkpoint root (validation only)')
    args = ap.parse_args()
    out = os.path.join(RES, 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device)
    ae = load_model(args.ae, args.seed, args.epoch, dev, args.ckpt_root)[0]

    m = args.margin
    keep_1d = np.arange(m, SIZE - m)
    keep = (keep_1d[:, None] * SIZE + keep_1d[None, :]).reshape(-1)
    res = {'ae': args.ae, 'seed': args.seed, 'epoch': args.epoch, 'margin': m}
    maps = {}
    for lay, path in GRIDS.items():
        g = h5py.File(path, 'r')['train']
        true_images = f32(g['self_vision_no_agent'][0])  # (441, 16, 64, 3) A-1 alone at each point
        ti_t = torch.from_numpy(true_images).to(dev)
        errs = {k: [] for k in ('self_self', 'other_other', 'self_other', 'other_self')}
        hit_o, hit_s, n_pairs = 0, 0, 0
        cfg_maps = {}
        for o in keep:
            sv = g['self_vision'][o]                 # (441, ...) A-1 at every s, A-2 at o
            s_true = f32(g['self_vision_no_agent'][o])
            o_true = f32(g['other_vision_no_agent'][o])
            srec, orec = rec_all(ae, sv, dev)
            k = keep
            errs['self_self'].append(np.abs(s_true[k] - srec[k]).mean((1, 2, 3)))
            errs['other_other'].append(np.abs(o_true[k] - orec[k]).mean((1, 2, 3)))
            errs['self_other'].append(np.abs(s_true[k] - orec[k]).mean((1, 2, 3)))
            errs['other_self'].append(np.abs(o_true[k] - srec[k]).mean((1, 2, 3)))
            # error-map minimum location (descriptive): every 2nd self point, on the GPU
            ss = k[::2]
            eo = (ti_t[None] - torch.from_numpy(orec[ss]).to(dev)[:, None]).abs().mean((2, 3, 4))
            es = (ti_t[None] - torch.from_numpy(srec[ss]).to(dev)[:, None]).abs().mean((2, 3, 4))
            po = np.stack(np.divmod(eo.argmin(1).cpu().numpy(), SIZE), 1)
            ps = np.stack(np.divmod(es.argmin(1).cpu().numpy(), SIZE), 1)
            hit_o += int((np.abs(po - np.array(divmod(int(o), SIZE))).max(1) <= 1).sum())
            hit_s += int((np.abs(ps - np.stack(np.divmod(ss, SIZE), 1)).max(1) <= 1).sum())
            n_pairs += len(ss)
            for a1, a2 in CONFIGS:
                if idx(a2) == o:
                    s = idx(a1)
                    cfg_maps[(a1, a2)] = (np.abs(true_images - orec[s]).mean((1, 2, 3)).reshape(SIZE, SIZE),
                                          np.abs(true_images - srec[s]).mean((1, 2, 3)).reshape(SIZE, SIZE),
                                          orec[s], srec[s], f32(g['other_vision_no_agent'][o][s]),
                                          f32(g['self_vision'][o][s]))
        r = {k: {'mean': float(np.concatenate(v).mean()), 'std': float(np.concatenate(v).std()),
                 'n': int(np.concatenate(v).size)} for k, v in errs.items()}
        r['margin_other_self_minus_other_other'] = r['other_self']['mean'] - r['other_other']['mean']
        r['map_min_within1_of_A2_other_rec'] = hit_o / n_pairs
        r['map_min_within1_of_A1_self_rec'] = hit_s / n_pairs
        r['n_pairs_map'] = n_pairs
        res[lay] = r
        maps[lay] = cfg_maps
        print(lay, json.dumps(r), flush=True)

    # figure: rows = configs, cols = per layout [other_rec map, self_rec map, A-2 true view, other_rec]
    arrays = {}
    fig, axes = plt.subplots(len(CONFIGS), 8, figsize=(26, 2.9 * len(CONFIGS)),
                             gridspec_kw={'width_ratios': [1, 1, 2.2, 2.2] * 2})
    for j, lay in enumerate(GRIDS):
        for i, (a1, a2) in enumerate(CONFIGS):
            emo, ems, orec, srec, otrue, sin = maps[lay][(a1, a2)]
            arrays[f'{lay}_A1{a1}_A2{a2}_errmap_other'] = emo
            arrays[f'{lay}_A1{a1}_A2{a2}_errmap_self'] = ems
            arrays[f'{lay}_A1{a1}_A2{a2}_other_rec'] = orec
            arrays[f'{lay}_A1{a1}_A2{a2}_A2_true'] = otrue
            for c, (em, title) in enumerate(((emo, 'other_rec vs all views'), (ems, 'self_rec vs all views'))):
                ax = axes[i, 4 * j + c]
                ax.imshow(em.T, origin='lower', extent=[-10.5, 10.5, -10.5, 10.5], cmap='jet')
                ax.plot(*a1, 'o', mfc='none', mec='w', ms=12, mew=2); ax.text(a1[0], a1[1], '1', color='w', ha='center', va='center', fontsize=8)
                ax.plot(*a2, 's', mfc='none', mec='w', ms=12, mew=2); ax.text(a2[0], a2[1], '2', color='w', ha='center', va='center', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f'{lay}: {title}', fontsize=8)
            axes[i, 4 * j + 2].imshow(otrue, aspect='auto', interpolation='nearest'); axes[i, 4 * j + 2].set_title('A-2 true view (no agent)', fontsize=8)
            axes[i, 4 * j + 3].imshow(np.clip(orec, 0, 1), aspect='auto'); axes[i, 4 * j + 3].set_title(f'A-2 view via Fig.5 (A-1 {a1}, A-2 {a2})', fontsize=8)
            for c in (2, 3):
                axes[i, 4 * j + c].set_xticks([]); axes[i, 4 * j + c].set_yticks([])
    fig.suptitle(f'v11 Fig.5 error maps: {args.ae} s{args.seed} ep{args.epoch} (1 = A-1, 2 = A-2; jet: low = blue). '
                 f'Array channels shown as RGB (v10 convention)', fontsize=10)
    fig.tight_layout()
    tag = f'{args.ae}_s{args.seed}_ep{args.epoch}'
    fig.savefig(os.path.join(out, f'fig5_maps_{tag}.png'), dpi=100)
    plt.close(fig)
    np.savez_compressed(os.path.join(out, f'fig5_maps_{tag}.npz'), **arrays)
    with open(os.path.join(out, f'fig5_{tag}.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
