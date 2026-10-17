"""
v11 prediction-image panels (instructions Sec.7.1; docs/v11_experiment_log.md
Sec.7.18 item 14). For one seed: both conditions x both layouts, on the same
viz20 episode numbers, t = 1, 5, 10, 20, 50.

Rows (columns = t):
  self input v_t | self truth v_{t+1} | self prediction | |error|
  self reconstruction (Fig.5 Autoencoder: self-encoder(v_t) -> decoder) | |error vs v_t|
  A-2 true view v2_t | A-2 viewpoint via Fig.5 (other-encoder(v_t) -> decoder) | |error|
Data: --data common -> v11_s1_random_viz20 for both models (identical frames);
      --data own    -> each condition's own viz20 (same episode numbers).
Without --ae_fmt the two Fig.5 rows are omitted (A-2 true view still shown).
Writes PNGs, one .npz with every array shown and a JSON of mean errors under
data/result/v11/<sub>/. Images: array channels shown as RGB (v10 convention).
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
from eval_trial_v11 import DATA, LAYNAME, RES, load_model, run  # noqa
import util  # noqa
import model.util as model_util  # noqa
from v11core.pleasure import pleasure_np  # noqa

_MASK_LOG = []
_orig_mask = model_util.mask


def _logged_mask(x, p_mask):
    # identical to model/util.mask (same torch.rand call -> same masks), but records
    # which batch elements were masked; calls alternate self, other per step.
    m = torch.rand((x.size(0), 1), device=x.device) < p_mask
    _MASK_LOG.append(m[:, 0].cpu().numpy())
    return torch.where(m, torch.zeros_like(x), x)

T_IMG = (1, 5, 10, 20, 50)


@torch.no_grad()
def ae_rec(ae, sv_u8, device):
    """sv_u8: (T, 16, 64, 3) uint8 -> self_rec, other_rec as uint8 (T, 16, 64, 3)."""
    x = torch.from_numpy(util.scale_vision(util.transpose_vision(sv_u8[None]))[0]).to(device)
    rec = ae(x, decode_from_other=True)
    out = {}
    for k in ('self_vision', 'other_vision'):
        r = ((rec[k] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        out[k] = np.round(r * 255).astype(np.uint8)
    return out['self_vision'], out['other_vision']


def absdiff(a, b):
    return np.abs(a.astype(int) - b.astype(int)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', required=True, help='prediction model, e.g. v11_s1_{}_mse')
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--ae_fmt', default=None, help='Fig.5 Autoencoder exp, e.g. v11_fig5_{}_s0')
    ap.add_argument('--ae_epoch', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data', default='common', choices=['common', 'own'])
    ap.add_argument('--sub', default='pred_images')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    out = os.path.join(RES, 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True

    arrays, summary = {}, {}
    for cond in ('random', 'mixed'):
        model, p_mask = load_model(args.exp_fmt.format(cond), args.seed, args.epoch, dev)
        ae = load_model(args.ae_fmt.format(cond), args.seed, args.ae_epoch, dev)[0] if args.ae_fmt else None
        dname = 'v11_s1_random_viz20' if args.data == 'common' else f'v11_s1_{cond}_viz20'
        path = os.path.join(DATA, dname, 'data.h5')
        _MASK_LOG.clear()
        model_util.mask = _logged_mask
        d = run(model, p_mask, path, 'train', 0, dev, keep_pred=True)
        model_util.mask = _orig_mask
        g = h5py.File(path, 'r')['train']
        assert len(_MASK_LOG) == 200, len(_MASK_LOG)  # one batch (viz20 <= 50 episodes), 100 steps x (self, other)
        vis_masked = np.stack(_MASK_LOG[0::2], 1)        # (N, 100): self vision masked at input t
        sv_all, ov_all, lay = g['self_vision'][()], g['other_vision'][()], g['layout'][()]
        s = {'pred_err_all': float(d['err'].mean())}
        if ae is not None:
            se, oe = [], []
            for n in range(len(lay)):
                sr, orr = ae_rec(ae, sv_all[n, :100], dev)
                se.append(np.abs(sr / 255. - sv_all[n, :100] / 255.).mean())
                oe.append(np.abs(orr / 255. - ov_all[n, :100] / 255.).mean())
            s['self_rec_err_all'] = float(np.mean(se))
            s['fig5_other_err_all'] = float(np.mean(oe))
        summary[cond] = s
        for k in (0, 1):
            ep = int(np.where(lay == k)[0][0])
            sv, ov, pr = sv_all[ep], ov_all[ep], d['pred'][ep]
            rows = [('input v_t', lambda t: sv[t]), ('truth v_t+1', lambda t: sv[t + 1]),
                    ('prediction', lambda t: pr[t]), ('|pred - truth|', lambda t: absdiff(pr[t], sv[t + 1]))]
            if ae is not None:
                sr, orr = ae_rec(ae, sv[:100], dev)
                rows += [('self rec (Fig.5 AE)', lambda t: sr[t]), ('|self rec - v_t|', lambda t: absdiff(sr[t], sv[t])),
                         ('A-2 true view v2_t', lambda t: ov[t]), ('A-2 view via Fig.5', lambda t: orr[t]),
                         ('|Fig.5 - v2_t|', lambda t: absdiff(orr[t], ov[t]))]
                arrays[f'{cond}_{LAYNAME[k]}_ep{ep}_selfrec'] = sr
                arrays[f'{cond}_{LAYNAME[k]}_ep{ep}_fig5other'] = orr
            else:
                rows += [('A-2 true view v2_t', lambda t: ov[t])]
            P_ep = pleasure_np(sv[:100])
            arrays[f'{cond}_{LAYNAME[k]}_ep{ep}_vision_masked'] = vis_masked[ep]
            arrays[f'{cond}_{LAYNAME[k]}_ep{ep}_pleasure'] = P_ep
            fig, axes = plt.subplots(len(rows), len(T_IMG), figsize=(3.2 * len(T_IMG), 1.05 * len(rows) + 1.0))
            for j, t in enumerate(T_IMG):
                for i, (name, fn) in enumerate(rows):
                    axes[i, j].imshow(fn(t), aspect='auto', interpolation='nearest')
                    axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                    if j == 0:
                        axes[i, j].set_ylabel(name, fontsize=7)
                axes[0, j].set_title(f't={t}  vision {"MASKED" if vis_masked[ep, t] else "SHOWN"}\n'
                                     f'P(bins 0-7) = ' + ' '.join(f'{x:+.2f}' for x in P_ep[t]), fontsize=7)
            fig.suptitle(f'{args.exp_fmt.format(cond)} s{args.seed} ep{args.epoch}; Fig.5 AE '
                         f'{args.ae_fmt.format(cond) if args.ae_fmt else "-"}; {dname} episode {ep} ({LAYNAME[k]}). '
                         f'Array channels shown as RGB (v10 convention)', fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'pred_{args.data}_{cond}_{LAYNAME[k]}_ep{ep}_s{args.seed}_e{args.epoch}.png'), dpi=110)
            plt.close(fig)
            arrays[f'{cond}_{LAYNAME[k]}_ep{ep}_self'] = sv
            arrays[f'{cond}_{LAYNAME[k]}_ep{ep}_other'] = ov
            arrays[f'{cond}_{LAYNAME[k]}_ep{ep}_pred'] = pr
    tag = f'{args.data}_s{args.seed}_e{args.epoch}'
    np.savez_compressed(os.path.join(out, f'pred_images_{tag}.npz'), **arrays)
    with open(os.path.join(out, f'pred_images_{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
