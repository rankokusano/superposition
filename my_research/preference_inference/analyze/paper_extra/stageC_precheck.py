"""
Paper round 14 (stage C, G diagnosis), Sec.1 checks (docs/paper_extra_log.md Sec.38; CPU only, no training).
  A  reproduction: the existing v12 stage-2 evaluation (analyze/v12/s2run.py + analyze/v11/eval_s2_v11.run_q2, unchanged,
     imported) of G v12_s2_pref_<c>_s<s> at ep200 is rerun on the CPU; per-episode sT2_t50 (green / swap sets) and
     dT5 (green / red sets, the saved T5 pairs) are compared with the saved GPU values (stage2/eval/pref_<c>_s<s>_ep200_per.npz).
  B  |Q2| on the preference-mixed test set (v11_s2_pref test, 300) for the primary G (pref, mixed / random, s0-s3,
     ep160-200): per episode x frame (t = 1..100 -> frames 1..100) norm, mean and quantiles; per bin group
     (bins 0,1,6,7 vs 2-5) mean |Q2| and variance share; projection on c+ (q_e, frames 1-50).
  C  epoch-0 G: the 00000.pth of each primary G vs its 00010.pth (G keys must differ) and vs the stage-I checkpoint
     it was built on (all shared non-G keys must be equal).
-> data/result/paper_extra/StageC/precheck/summary.json
"""

import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_PI, 'analyze', 'v12'), os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)
import s2run  # noqa
import eval_s2_v11 as E  # noqa
from eval_trial_v12 import RES, load_v12  # noqa
import h5py  # noqa

OUT = os.path.join(RES, 'paper_extra', 'StageC', 'precheck')
EV = os.path.join(RES, 'v12', 'stage2', 'eval')
DEV = torch.device('cpu')
CP = np.array(json.load(open(os.path.join(RES, 'v14', 'cand_shape.json')))['c_plus'], np.float64)


def q2(exp, s, e, setname):
    m, _ = load_v12(exp, s, e, DEV)
    f = s2run.S2Feed(m)
    d = E.load_set(setname)
    f.reset(h5py.File(d['path'], 'r')[d['split']]['self_motion'][()])
    Q, _ = E.run_q2(f, d['path'], d['split'], DEV)
    return Q.astype(np.float64), d


def gsel(q, idx):
    return np.take_along_axis(q, idx, 1).mean(1)


def part_a(R, c, s):
    t0 = time.time()
    Qg, dg = q2(f'v12_s2_pref_{c}_s{s}', s, 200, 'green')
    Qs, ds = q2(f'v12_s2_pref_{c}_s{s}', s, 200, 'swap')
    Qr, _ = q2(f'v12_s2_pref_{c}_s{s}', s, 200, 'red')
    L0 = ds['layout']
    newG = np.array([E.RB[int(l)] for l in L0]); oldG = np.array([E.GB[int(l)] for l in L0])
    sT2 = (gsel(Qs[:, 50], newG) - gsel(Qs[:, 50], oldG)) - (gsel(Qg[:, 50], newG) - gsel(Qg[:, 50], oldG))
    pairs = np.load(os.path.join(EV, 't5_pairs.npz'))['pairs']
    Lp = pairs[:, 0]

    def gr(q, L):
        return gsel(q, np.array([E.GB[int(l)] for l in L])) - gsel(q, np.array([E.RB[int(l)] for l in L]))
    dT5 = gr(Qg[pairs[:, 1], pairs[:, 2]], Lp) - gr(Qr[pairs[:, 3], pairs[:, 4]], Lp)
    sv = np.load(os.path.join(EV, f'pref_{c}_s{s}_ep200_per.npz'))
    R['A'][f'pref_{c}_s{s}_ep200'] = {
        'sT2_bit_identical': bool(np.array_equal(sT2.astype(sv['sT2_t50'].dtype), sv['sT2_t50'])),
        'sT2_max_abs_diff': float(np.abs(sT2 - sv['sT2_t50']).max()), 'sT2_mean_cpu': float(sT2.mean()),
        'sT2_mean_saved': float(sv['sT2_t50'].mean()),
        'dT5_bit_identical': bool(np.array_equal(dT5.astype(sv['dT5'].dtype), sv['dT5'])),
        'dT5_max_abs_diff': float(np.abs(dT5 - sv['dT5']).max()), 'dT5_mean_cpu': float(dT5.mean()),
        'dT5_mean_saved': float(sv['dT5'].mean()), 'seconds': time.time() - t0}
    print('A', c, s, R['A'][f'pref_{c}_s{s}_ep200'], flush=True)


def part_b(R, c, s, e):
    Q, d = q2(f'v12_s2_pref_{c}_s{s}', s, e, 'pref')        # (300, 100, 8); step t -> frame t + 1
    n = np.linalg.norm(Q, axis=-1)
    a, b = Q[..., [0, 1, 6, 7]], Q[..., 2:6]
    var = Q.reshape(-1, 8).var(0)
    qe = (Q[:, :50] @ (CP / np.linalg.norm(CP))).mean(1)
    R['B'][f'pref_{c}_s{s}_ep{e}'] = {
        'norm_mean': float(n.mean()), 'norm_q': {str(q): float(np.percentile(n, q)) for q in (5, 25, 50, 75, 95)},
        'norm_bins0167_mean': float(np.linalg.norm(a, axis=-1).mean()), 'norm_bins2to5_mean': float(np.linalg.norm(b, axis=-1).mean()),
        'abs_mean_per_bin': np.abs(Q).mean((0, 1)).tolist(), 'var_share_bins2to5': float(var[2:6].sum() / var.sum()),
        'q_e_frames1_50': {'mean': float(qe.mean()), 'sd': float(qe.std()), 'abs_mean': float(np.abs(qe).mean())}}


def part_c(R):
    for c, s1 in (('mixed', 'v12_s1_mixed_l1_s{s}'), ('random', 'v12_s1_random_l1_s{s}')):
        for s in range(4):
            g0 = torch.load(os.path.join(RES, f'v12_s2_pref_{c}_s{s}', str(s), 'model', '00000.pth'), map_location='cpu')['model']
            g1 = torch.load(os.path.join(RES, f'v12_s2_pref_{c}_s{s}', str(s), 'model', '00010.pth'), map_location='cpu')['model']
            st = torch.load(os.path.join(RES, s1.format(s=s), str(s), 'model', '00200.pth'), map_location='cpu')['model']
            gk = [k for k in g0 if k not in st]
            shared = [k for k in g0 if k in st]
            R['C'][f'pref_{c}_s{s}'] = {
                'n_keys': len(g0), 'n_G_only_keys': len(gk), 'G_only_keys_sample': gk[:6],
                'G_keys_changed_00000_to_00010': int(sum(not torch.equal(g0[k], g1[k]) for k in gk)),
                'shared_keys_equal_to_stage1_ep200': f'{sum(torch.equal(g0[k], st[k]) for k in shared)}/{len(shared)}',
                'shared_keys_equal_00000_00010': f'{sum(torch.equal(g0[k], g1[k]) for k in shared)}/{len(shared)}'}
            print('C', c, s, R['C'][f'pref_{c}_s{s}'], flush=True)


def main():
    torch.set_num_threads(int(os.environ.get('NT', '8')))
    os.makedirs(OUT, exist_ok=True)
    R = {'A': {}, 'B': {}, 'C': {}}
    part_c(R)
    for c in ('mixed', 'random'):
        part_a(R, c, 0)
    for c in ('mixed', 'random'):
        for s in range(4):
            for e in (160, 170, 180, 190, 200):
                part_b(R, c, s, e)
        print('B done', c, flush=True)
    json.dump(R, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
