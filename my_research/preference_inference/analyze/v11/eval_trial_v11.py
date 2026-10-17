"""
v11 trial (6.2) evaluation (docs/v11_experiment_log.md Sec.5.2, 7.3).
Reads checkpoints/data only; writes under data/result/v11/trial/.

For each model (condition x seed x epoch) it runs the model step by step
exactly as the runner does (model.eval(), init_state, p_mask 0 at t=0 and
p_mask_vision afterwards, util.seed_all(eval_seed) right before each pass),
and collects h^1, the one-step prediction and positions.

  R^2 h1->self   (a) v4 convention: h after input frame t <-> p_t, Ridge(1),
                     in-sample on the model's own training data (first 1000);
                 (b) same pairing, fit on random-train h, held-out on the
                     common random test;
                 (c) M1 pairing (h after frame t-1 <-> p_t), (b)-style.
  M1(a) simple   regressor (c) fit on h from v11_s1_random train (first
                 1000) through this model; on v11_s1_random test:
                 b_t = (pos_hat - pos) . unit(red_ref - pos), frames within 2
                 of red_ref excluded; per-episode mean over t; mean + 95% CI
                 (episode bootstrap 1000, seed 0); per-layout; per-t curve.
  M3 simple      per-pixel L1 of the one-step prediction (in [0,1]) on the
                 own test and the common random test, per layout.
  aux            red pixel count of the one-step prediction minus that of
                 the true next frame (common random test), per layout.
  images         viz20 (same episode numbers in both conditions): episode 0
                 (original) and the first swap episode, t = 1,5,10,20,50:
                 input v_t, truth v_{t+1}, prediction, |error|, A-2's true
                 view (no Fig.5 decoder in the trial).
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
from sklearn.linear_model import Ridge

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
import util  # noqa
from config_util import load_config  # noqa
from model import model_v11  # noqa
from sim_v10 import layout as L  # noqa
from v11core.pleasure import color_mask  # noqa

DATA = os.path.join(_PI_DIR, 'data', 'data')
RES = os.path.join(_PI_DIR, 'data', 'result')
RED_REF = {0: L.LANDMARKS[L.ORIGINAL]['Red'], 1: L.LANDMARKS[L.SWAP_RG]['Red']}
LAYNAME = {0: 'original', 1: 'swap_rg'}
T_IMG = (1, 5, 10, 20, 50)


def load_model(exp, seed, epoch, device):
    cfg = load_config(os.path.join(_PI_DIR, 'config', 'exp', exp + '.yml'))
    mcfg = load_config(os.path.join(_PI_DIR, 'config', 'model', cfg.model.name, cfg.model.config + '.yml'))
    m = getattr(model_v11, cfg.model.name)(mcfg).to(device)
    ck = os.path.join(RES, exp, str(seed), 'model', f'{epoch:05d}.pth')
    m.load_state_dict(torch.load(ck, map_location='cpu')['model'])
    m.eval()
    return m, cfg.p_mask_vision


@torch.no_grad()
def run(model, p_mask, h5path, split, n_ep, device, eval_seed=0, batch=50, keep_pred=False):
    g = h5py.File(h5path, 'r')[split]
    n = min(n_ep, g['layout'].shape[0]) if n_ep else g['layout'].shape[0]
    util.seed_all(eval_seed)
    H, P, ERR, RED_D, RED_T = [], [], [], [], []
    PRED, HID = [], []
    for b0 in range(0, n, batch):
        sv = g['self_vision'][b0:b0 + batch]  # (B, 101, 16, 64, 3) uint8
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(sv))).to(device)
        B = x.shape[0]
        model.init_state(B)
        hs, errs, redd, redt, preds = [], [], [], [], []
        for t in range(100):
            pm = 0 if t == 0 else p_mask
            pred = model({'self_vision': x[:, t]}, pm, pm)['self_vision']
            hs.append(model.superposition_module.state['self'][0].cpu().numpy())
            p01 = ((pred + 1) / 2).clamp(0, 1)
            tr01 = (x[:, t + 1] + 1) / 2
            errs.append((p01 - tr01).abs().mean((1, 2, 3)).cpu().numpy())
            pr = p01.permute(0, 2, 3, 1).cpu().numpy()
            rt = color_mask(sv[:, t + 1], 'Red').sum((1, 2))
            redd.append(color_mask(pr, 'Red').sum((1, 2)) - rt)
            redt.append(rt)
            if keep_pred == 'f16':
                preds.append(pr.astype(np.float16))
            elif keep_pred:
                preds.append(np.round(pr * 255).astype(np.uint8))
        H.append(np.stack(hs, 1))
        ERR.append(np.stack(errs, 1))
        RED_D.append(np.stack(redd, 1))
        RED_T.append(np.stack(redt, 1))
        if keep_pred:
            PRED.append(np.stack(preds, 1))
        if hasattr(model, '_hidden'):
            HID.append(model._hidden[:, :100].numpy() & bool(getattr(model, 'pleasure_masking', False)))
    out = {
        'h': np.concatenate(H),            # (N, 100, 128): h after input frame t
        'err': np.concatenate(ERR),        # (N, 100)
        'red_diff': np.concatenate(RED_D).astype(np.float32),
        'red_truth': np.concatenate(RED_T).astype(np.float32),
        'pos': g['self_position'][:n],     # (N, 101, 2)
        'layout': g['layout'][:n],
    }
    if keep_pred:
        out['pred'] = np.concatenate(PRED)
    out['hidden'] = np.concatenate(HID) if HID else np.zeros(out['err'].shape, bool)
    return out


def r2(y, yhat):
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def boot_ci(v, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), (n, len(v)))
    m = v[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def m1a(reg, d, color='Red'):
    """M1 pairing: h after frame t-1 (index t-1) <-> p_t, t = 1..100.
    color != 'Red' gives the same projection toward another landmark's
    reference point (control for regressor shrinkage, not pre-registered)."""
    h = d['h'].reshape(-1, 128)
    pos = d['pos'][:, 1:101]
    est = reg.predict(h).reshape(pos.shape)
    red = np.stack([L.LANDMARKS[LAYNAME[int(l)]][color] for l in d['layout']])[:, None, :]
    u = red - pos
    dist = np.linalg.norm(u, axis=-1)
    u = u / np.maximum(dist[..., None], 1e-9)
    b = ((est - pos) * u).sum(-1)
    valid = dist >= 2
    ep_mean = np.array([b[i][valid[i]].mean() for i in range(len(b))])
    curve = np.array([b[:, t][valid[:, t]].mean() for t in range(100)])
    return b, valid, ep_mean, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conds', nargs='+', default=['random', 'mixed'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epoch', type=int, default=60)
    ap.add_argument('--exp_fmt', default='v11_trial_{}_mse')
    ap.add_argument('--n_train', type=int, default=1000)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default=os.path.join(RES, 'v11', 'trial'))
    args = ap.parse_args()
    assert os.path.realpath(args.out).startswith(os.path.realpath(os.path.join(RES, 'v11')) + os.sep)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rnd = os.path.join(DATA, 'v11_s1_random', 'data.h5')
    summary, ep_b, img, ep_contrast = {}, {}, {}, {}
    for c in args.conds:
        exp = args.exp_fmt.format(c)
        model, p_mask = load_model(exp, args.seed, args.epoch, dev)
        own = os.path.join(DATA, f'v11_s1_{c}', 'data.h5')
        d_rtr = run(model, p_mask, rnd, 'train', args.n_train, dev)
        d_rte = run(model, p_mask, rnd, 'test', 0, dev)
        d_own_tr = d_rtr if c == 'random' else run(model, p_mask, own, 'train', args.n_train, dev)
        d_own_te = d_rte if c == 'random' else run(model, p_mask, own, 'test', 0, dev)
        # bit-identity of the eval pass
        d_chk = run(model, p_mask, rnd, 'test', 0, dev)
        ident = bool(np.array_equal(d_chk['h'], d_rte['h']) and np.array_equal(d_chk['err'], d_rte['err']))

        s = {'exp': exp, 'seed': args.seed, 'epoch': args.epoch, 'p_mask_vision': p_mask,
             'eval_seed': 0, 'eval_bit_identical_rerun_20ep': ident}
        # R^2, v4 convention: h after frame t (index t) <-> p_t, t = 0..99
        h_own = d_own_tr['h'].reshape(-1, 128)
        p_own = d_own_tr['pos'][:, :100].reshape(-1, 2)
        s['r2_v4conv_insample_own_train'] = r2(p_own, Ridge(alpha=1.0).fit(h_own, p_own).predict(h_own))
        reg_v4 = Ridge(alpha=1.0).fit(d_rtr['h'].reshape(-1, 128), d_rtr['pos'][:, :100].reshape(-1, 2))
        s['r2_v4conv_fit_random_train_on_random_test'] = r2(
            d_rte['pos'][:, :100].reshape(-1, 2), reg_v4.predict(d_rte['h'].reshape(-1, 128)))
        # M1 pairing regressor
        reg = Ridge(alpha=1.0).fit(d_rtr['h'].reshape(-1, 128), d_rtr['pos'][:, 1:101].reshape(-1, 2))
        s['r2_m1pair_fit_random_train_on_random_test'] = r2(
            d_rte['pos'][:, 1:101].reshape(-1, 2), reg.predict(d_rte['h'].reshape(-1, 128)))
        b, valid, ep_mean, curve = m1a(reg, d_rte)
        ep_b[c] = ep_mean
        ctrl = {}
        for col in ('Green', 'Blue', 'Cyan'):
            _, _, em, _ = m1a(reg, d_rte, col)
            ctrl[col] = em
        s['M1a_control_other_corners_NOT_preregistered'] = {
            col: {'mean': float(v.mean()), 'ci95': boot_ci(v)} for col, v in ctrl.items()}
        s['M1a_control_other_corners_NOT_preregistered']['red_minus_mean_other3'] = {
            'mean': float((ep_mean - np.mean(list(ctrl.values()), 0)).mean()),
            'ci95': boot_ci(ep_mean - np.mean(list(ctrl.values()), 0))}
        s['M1a_control_other_corners_NOT_preregistered']['red_minus_green'] = {
            'mean': float((ep_mean - ctrl['Green']).mean()), 'ci95': boot_ci(ep_mean - ctrl['Green'])}
        ep_contrast[c] = ep_mean - ctrl['Green']
        s['M1a'] = {'mean': float(ep_mean.mean()), 'ci95': boot_ci(ep_mean),
                    'per_layout': {LAYNAME[k]: {'mean': float(ep_mean[d_rte['layout'] == k].mean()),
                                                'ci95': boot_ci(ep_mean[d_rte['layout'] == k])} for k in (0, 1)},
                    'curve_t1_100': curve.round(4).tolist()}
        # M3
        s['M3_l1_per_pixel'] = {
            'common_random_test': {LAYNAME[k]: float(d_rte['err'][d_rte['layout'] == k].mean()) for k in (0, 1)},
            'own_test': {LAYNAME[k]: float(d_own_te['err'][d_own_te['layout'] == k].mean()) for k in (0, 1)},
        }
        s['M3_l1_per_pixel']['common_random_test']['all'] = float(d_rte['err'].mean())
        s['M3_l1_per_pixel']['own_test']['all'] = float(d_own_te['err'].mean())
        # aux: red pixel count difference (pred - truth), common random test
        rd = d_rte['red_diff'].mean(1)
        s['aux_red_pixels_pred_minus_truth'] = {
            'mean_per_frame': float(rd.mean()), 'ci95': boot_ci(rd),
            'per_layout': {LAYNAME[k]: {'mean': float(rd[d_rte['layout'] == k].mean()),
                                        'ci95': boot_ci(rd[d_rte['layout'] == k])} for k in (0, 1)},
            'truth_red_pixels_mean': float(d_rte['red_truth'].mean()),
        }
        summary[c] = s
        # images
        viz = os.path.join(DATA, f'v11_s1_{c}_viz20', 'data.h5')
        d_viz = run(model, p_mask, viz, 'train', 0, dev, keep_pred=True)
        gv = h5py.File(viz, 'r')['train']
        img[c] = {'pred': d_viz['pred'], 'sv': gv['self_vision'][()], 'ov': gv['other_vision'][()],
                  'layout': gv['layout'][()], 'err': d_viz['err']}
        print(c, json.dumps({k: v for k, v in s.items() if k != 'M1a'}), 'M1a', s['M1a']['mean'], s['M1a']['ci95'], flush=True)

    if 'random' in ep_b and 'mixed' in ep_b:
        diff = ep_b['mixed'] - ep_b['random']
        summary['M1a_paired_diff_mixed_minus_random'] = {'mean': float(diff.mean()), 'ci95': boot_ci(diff)}
        dc = ep_contrast['mixed'] - ep_contrast['random']
        summary['M1a_red_minus_green_paired_diff_NOT_preregistered'] = {'mean': float(dc.mean()), 'ci95': boot_ci(dc)}
    with open(os.path.join(args.out, f'trial_eval_s{args.seed}_ep{args.epoch}.json'), 'w') as f:
        json.dump(summary, f, indent=1)

    # M1(a) curve figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for c in args.conds:
        ax[0].plot(range(1, 101), summary[c]['M1a']['curve_t1_100'], label=c)
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set_xlabel('t'); ax[0].set_ylabel('bias toward red (arena units)')
    ax[0].set_title('M1(a) simple: (pos_hat - pos) . u_red, common random test'); ax[0].legend()
    labels, means, lo, hi = [], [], [], []
    for c in args.conds:
        for k in ('original', 'swap_rg'):
            v = summary[c]['M1a']['per_layout'][k]
            labels.append(f'{c}\n{k}'); means.append(v['mean']); lo.append(v['ci95'][0]); hi.append(v['ci95'][1])
    means = np.array(means)
    ax[1].bar(range(len(means)), means, yerr=[means - np.array(lo), np.array(hi) - means], capsize=4,
              color=['tab:gray', 'tab:gray', 'tab:orange', 'tab:orange'][:len(means)])
    ax[1].axhline(0, color='k', lw=0.5)
    ax[1].set_xticks(range(len(means))); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_title('episode mean, 95% CI')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f'trial_M1a_s{args.seed}_ep{args.epoch}.png'), dpi=130)
    plt.close(fig)

    # prediction images (Sec.7.1 self rows + A-2's true view)
    arrays = {}
    for c in args.conds:
        lay = img[c]['layout']
        for k in (0, 1):
            ep = int(np.where(lay == k)[0][0])
            sv, pr, ov = img[c]['sv'][ep], img[c]['pred'][ep], img[c]['ov'][ep]
            fig, axes = plt.subplots(5, len(T_IMG), figsize=(3.2 * len(T_IMG), 5.2))
            rows = ['input v_t', 'truth v_t+1', 'prediction', '|error|', 'A-2 true view v2_t+1']
            for j, t in enumerate(T_IMG):
                ims = [sv[t], sv[t + 1], pr[t], np.abs(pr[t].astype(int) - sv[t + 1].astype(int)).astype(np.uint8), ov[t + 1]]
                for i, im in enumerate(ims):
                    axes[i, j].imshow(im, aspect='auto', interpolation='nearest')
                    axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                    if j == 0:
                        axes[i, j].set_ylabel(rows[i], fontsize=8)
                axes[0, j].set_title(f't={t}', fontsize=9)
            fig.suptitle(f'trial {c} s{args.seed} ep{args.epoch}, viz20 episode {ep} ({LAYNAME[k]}); '
                         f'mean |pred-truth| (this ep) {img[c]["err"][ep].mean():.4f}. Array channels shown as RGB (v10 convention)',
                         fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f'trial_pred_{c}_{LAYNAME[k]}_ep{ep}_s{args.seed}_e{args.epoch}.png'), dpi=110)
            plt.close(fig)
            arrays[f'{c}_{LAYNAME[k]}_ep{ep}_pred'] = pr
            arrays[f'{c}_{LAYNAME[k]}_ep{ep}_self'] = sv
            arrays[f'{c}_{LAYNAME[k]}_ep{ep}_other'] = ov
    np.savez_compressed(os.path.join(args.out, f'trial_pred_images_s{args.seed}_ep{args.epoch}.npz'), **arrays)
    print(json.dumps(summary.get('M1a_paired_diff_mixed_minus_random')))


if __name__ == '__main__':
    main()
