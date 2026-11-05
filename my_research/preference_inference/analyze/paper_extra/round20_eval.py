"""
Paper round 20 Sec.2 evaluation (paper/check/paper_instructions_round20.md; docs/paper_extra_log.md Sec.44; exploratory,
outside the pre-registration; procedure committed before computing). No training; predictors and G fixed. GPU.
Data: data/data/paper_r20_swap_p{80,50}/data_{N,O,S,R}.h5 (collect_data_round20.py; 600 episodes, frames 0..80).
Flow (stage B / stage C): model.eval(), util.seed_all(0) at the start of every run, cudnn deterministic, batch 50,
A-1's true motion fed, process-1 pleasure from the current (unmasked) view; step t = 0..79 takes frame t and predicts
frame t + 1. Vision shown (mask p 0, both processes) at the steps of the vision mode, hidden (p 1) otherwise:
  V0 {0}  (the existing evaluation)      V1 {0, 50, 51, 52}  (main)      V2 all steps
Process-2 input:
  T  [0, 0, P^A2_t, 1], P^A2_t = A-2's true pleasure (liked - disliked, pleasure_np) from A-2's actual view at frame t
  Z  [0, 0, 0 x 8, 1]   (the c0 candidate)
  G  G's own output (stage-C flow; G reads the unmasked other-encoder output of A-1's view at every step)
Per run and episode, per step: FP = mse(predict_feature(h2_t), other_vision_encoder(x_{t+1})) (as eval_inverse_v14.run),
L1 = |pred - x_{t+1}| per pixel on the [-1, 1] scale (as eval_inverse_v14.run), and for V1 / V2 the self-position
tracking error: Ridge(1) h1_t -> A-1 position at frame t + 1 (the G1 read-out of eval_trial_v12.tracking), fitted on
the no-swap variants (S and R pooled, all steps) with episode GroupKFold(5), applied to the held-out episodes of every
variant (A-1's path is the same in all variants); error = Euclidean distance.
Families (checkpoint -> what is stored):
  mse  v12mse:mixed / v12mse:random / csr50mse:signrand50, pairs 0-5, ep160-200 : FP (V0 V1 V2), tracking (V1 V2); T, Z
  l1   v12:mixed / v12:random / csr50:signrand50, pairs 0-5, ep160-200         : L1 (V1 V2); T, Z
  G    v12_s2_pref_{mixed,random}_s{0..3} ep160-200, pre-training ep0 of both  : FP (V0 V1 V2), L1, tracking (V1 V2)
-> data/result/paper_extra/Round20/eval/<fam>_<cond>_s<s>_ep<e>.npz
   keys FP|L1|TRK _ p80|p50 _ V0|V1|V2 _ T|Z|G _ N|O|S|R, (600, 80) float32
Modes: --check (Sec.1.2 check 4, GPU1, first 50 episodes), --jobs <fam:cond:s:e> ... (full evaluation).
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import vdir_c_run as VC  # noqa  (eval_inverse_v14 with the csr50 loader)
import stageC_eval as SC  # noqa
from eval_trial_v12 import RES, load_v12  # noqa
import util  # noqa
from v11core.pleasure import pleasure_np  # noqa

EI = VC.mod
DATA = os.path.join(_PI, 'data', 'data')
OUT = os.path.join(RES, 'paper_extra', 'Round20')
PS = (80, 50)
VARS = ('N', 'O', 'S', 'R')
T_S, NSTEP = 50, 80
VMODES = {'V0': (0,), 'V1': (0, T_S, T_S + 1, T_S + 2), 'V2': tuple(range(NSTEP))}
TZ_SPEC = {'mse': {'p50': 'v12mse:mixed', 'none': 'v12mse:random', 'sr50': 'csr50mse:signrand50'},
           'l1': {'p50': 'v12:mixed', 'none': 'v12:random', 'sr50': 'csr50:signrand50'}}
G_EXP = {'p50': 'v12_s2_pref_mixed', 'none': 'v12_s2_pref_random', 'pre': 'v12_s2_pref_mixed', 'pre_none': 'v12_s2_pref_random'}


def dpath(p, v):
    return os.path.join(DATA, f'paper_r20_swap_p{p}', f'data_{v}.h5')


def a2_pleasure(ov, liked):
    out = np.empty(ov.shape[:2] + (8,), np.float32)
    for lk in (0, 1):
        m = liked == lk
        a, b = ('Red', 'Green') if lk == 1 else ('Green', 'Red')
        out[m] = pleasure_np(ov[m], a, b)
    return out


def load_data(p, n_ep=0, path=None):
    """-> {var: {'sv': uint8 (N, 81, 16, 64, 3), 'sm', 'pos' (A-1), 'PT' (N, 80, 8)}}"""
    D = {}
    for v in VARS:
        g = h5py.File(path(p, v) if path else dpath(p, v), 'r')['test']
        N = n_ep or g['self_vision'].shape[0]
        lk = g['a2_liked'][:N]
        D[v] = {'sv': g['self_vision'][:N], 'sm': g['self_motion'][:N], 'pos': g['self_position'][:N].astype(np.float64),
                'PT': a2_pleasure(g['other_vision'][:N, :NSTEP], lk), 'liked': lk, 'layout': g['layout'][:N]}
    return D


@torch.no_grad()
def run(model, sv, sm, P8, shown, dev, nstep=NSTEP, want_h1=False, want_q=False, batch=50):
    """sv uint8 (N, T, 16, 64, 3), sm (N, T, 2), P8 (N, nstep, 8) or None (G's own output), shown: steps with vision."""
    util.seed_all(0)
    N = sv.shape[0]
    FP, L1, H1, Q = [], [], [], []
    for b0 in range(0, N, batch):
        b1 = min(b0 + batch, N)
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(sv[b0:b1]))).to(dev)
        m = torch.from_numpy(sm[b0:b1]).float().to(dev)
        B = x.shape[0]
        model.init_state(B)
        fp, l1, h1, q = [], [], [], []
        for t in range(nstep):
            pm = 0 if t in shown else 1
            if P8 is None:
                p2 = None
            else:
                c = torch.from_numpy(P8[b0:b1, t]).float().to(dev)
                z = torch.zeros(B, 1, device=dev)
                p2 = torch.cat([z, z, c, torch.ones_like(z)], 1)
            o = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=p2)
            l1.append(torch.abs(o['self_vision'] - x[:, t + 1]).mean((1, 2, 3)).cpu().numpy())
            os_ = model.get_state()['other'][0]
            fp.append(F.mse_loss(model.predict_feature(os_), model.other_vision_encoder_module(x[:, t + 1]),
                                 reduction='none').mean(1).cpu().numpy())
            if want_h1:
                h1.append(model.superposition_module.state['self'][0].cpu().numpy())
            if want_q:
                q.append((o['q2_hat'].cpu().numpy(), os_.cpu().numpy()))
        FP.append(np.stack(fp, 1)); L1.append(np.stack(l1, 1))
        if want_h1:
            H1.append(np.stack(h1, 1))
        if want_q:
            Q.append((np.stack([a for a, _ in q], 1), np.stack([b for _, b in q], 1)))
    out = {'FP': np.concatenate(FP), 'L1': np.concatenate(L1)}
    if want_h1:
        out['h1'] = np.concatenate(H1)
    if want_q:
        out['Q'] = np.concatenate([a for a, _ in Q]); out['H'] = np.concatenate([b for _, b in Q])
    return out


def tracking(h1, pos):
    """h1 {var: (N, S, 128)}, pos (N, S+1, 2) (A-1, same for all variants) -> {var: (N, S) error}."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    N, S = h1['S'].shape[:2]
    Y = pos[:, 1:S + 1]
    out = {v: np.zeros((N, S), np.float32) for v in h1}
    for tr, te in GroupKFold(5).split(np.zeros(N), groups=np.arange(N)):
        Xtr = np.concatenate([h1['S'][tr].reshape(-1, 128), h1['R'][tr].reshape(-1, 128)]).astype(np.float64)
        Ytr = np.concatenate([Y[tr].reshape(-1, 2)] * 2)
        r = Ridge(alpha=1.0).fit(Xtr, Ytr)
        for v in h1:
            Yh = r.predict(h1[v][te].reshape(-1, 128).astype(np.float64)).reshape(len(te), S, 2)
            out[v][te] = np.linalg.norm(Yh - Y[te], axis=2)
    return out


def load_model(fam, cond, s, e, dev):
    if fam == 'G':
        m, _ = load_v12(f'{G_EXP[cond]}_s{s}', s, e, dev)
        m.eval()
        return m
    m, width = EI.load(f'{TZ_SPEC[fam][cond]}:{s}', e, dev)
    assert width == 11
    return m


def evaluate(fam, cond, s, e, dev, DD):
    model = load_model(fam, cond, s, e, dev)
    sav, t0 = {}, time.time()
    inputs = ('G',) if fam == 'G' else ('T', 'Z')
    vmodes = ('V1', 'V2') if fam == 'l1' else ('V0', 'V1', 'V2')
    for p in PS:
        D = DD[p]
        for vm in vmodes:
            for inp in inputs:
                h1 = {}
                for v in VARS:
                    P8 = None if inp == 'G' else (D[v]['PT'] if inp == 'T' else np.zeros_like(D[v]['PT']))
                    r = run(model, D[v]['sv'], D[v]['sm'], P8, VMODES[vm], dev, want_h1=(fam != 'l1' and vm != 'V0'))
                    k = f'{p}_{vm}_{inp}_{v}'
                    if fam in ('mse', 'G'):
                        sav[f'FP_p{k}'] = r['FP'].astype(np.float32)
                    if fam in ('l1', 'G'):
                        sav[f'L1_p{k}'] = r['L1'].astype(np.float32)
                    if 'h1' in r:
                        h1[v] = r['h1']
                if h1:
                    for v, err in tracking(h1, D['S']['pos']).items():
                        sav[f'TRK_p{p}_{vm}_{inp}_{v}'] = err
    del model
    torch.cuda.empty_cache()
    return sav, time.time() - t0


# ------------------------------------------------------------------ Sec.1.2 check 4
def pad_file(p, v, n, dst):
    """first n episodes of the new data, frames 0..80, padded to 101 frames by repeating frame 80 (for the
    existing code, which runs 100 steps; steps < 80 do not see the padding)."""
    g = h5py.File(dpath(p, v), 'r')['test']
    with h5py.File(dst, 'w') as f:
        h = f.create_group('test')
        for k in ('self_vision', 'other_vision', 'self_motion', 'self_position', 'other_position', 'layout_t'):
            a = g[k][:n]
            pad = np.repeat(a[:, -1:], 101 - a.shape[1], 1)
            h.create_dataset(k, data=np.concatenate([a, pad], 1))
        for k in ('layout', 'a2_liked'):
            h.create_dataset(k, data=g[k][:n])


def check(dev):
    n = 50
    od = os.path.join(OUT, 'check')
    os.makedirs(od, exist_ok=True)
    R = {'device': torch.cuda.get_device_name(dev), 'n_ep': n}
    D = load_data(80, n)
    # (a) existing eval_inverse_v14.run on the padded new data vs this code, V0, Z (= c0) and T (true pleasure)
    for spec, e in (('v12mse:mixed:0', 200), ('v12:mixed:0', 200), ('csr50mse:signrand50:0', 200)):
        model, width = EI.load(spec, e, dev)
        for v in ('N', 'O', 'S'):
            dst = os.path.join(od, f'padded_p80_{v}.h5')
            if not os.path.exists(dst):
                pad_file(80, v, n, dst)
            d = {'path': dst, 'split': 'test'}
            for inp in ('Z', 'T'):
                P8 = D[v]['PT'] if inp == 'T' else np.zeros_like(D[v]['PT'])
                P8pad = np.concatenate([P8, np.repeat(P8[:, -1:], 100 - NSTEP, 1)], 1)
                old = EI.run(model, width, d, P8pad, dev, n)
                new = run(model, D[v]['sv'], D[v]['sm'], P8, VMODES['V0'], dev)
                R[f'a|{spec}|ep{e}|{v}|{inp}'] = {
                    q: {'first_half_bit_identical': bool(np.array_equal(old[q][:, :T_S], new[q][:, :T_S])),
                        'all_80_bit_identical': bool(np.array_equal(old[q][:, :NSTEP], new[q])),
                        'max_abs_diff_80': float(np.abs(old[q][:, :NSTEP] - new[q]).max())} for q in ('FP', 'L1')}
        del model
    # (b) this code on the stage-B set (first 50 episodes, 100 steps, V0) vs the saved stage-B arrays
    gb = h5py.File(os.path.join(DATA, 'paper_vdir_b_pref', 'data.h5'), 'r')['test']
    svb, smb, ovb, lkb = gb['self_vision'][:n], gb['self_motion'][:n], gb['other_vision'][:n, :100], gb['a2_liked'][:n]
    PTb = a2_pleasure(ovb, lkb)
    for spec, e, tag in (('v12mse:mixed:0', 200, 'v12mse_mixed_0'), ('v12:mixed:0', 200, 'v12_mixed_0')):
        model, _ = EI.load(spec, e, dev)
        sv = np.load(os.path.join(RES, 'paper_extra', 'Vdir_B', 'eval', 'inverse', f'{tag}_ep{e}_vdirb.npz'))
        for inp in ('Z', 'T'):
            P8 = PTb if inp == 'T' else np.zeros_like(PTb)
            new = run(model, svb, smb, P8, VMODES['V0'], dev, nstep=100)
            if inp == 'Z':
                ref = {q: sv[f'{q}_c0'][:n] for q in ('FP', 'L1')}
            else:
                ref = {q: np.where((lkb == 1)[:, None], sv[f'{q}_ref_red'][:n], sv[f'{q}_ref_green'][:n]) for q in ('FP', 'L1')}
            R[f'b|{spec}|ep{e}|stageB|{inp}'] = {q: {'bit_identical': bool(np.array_equal(ref[q], new[q])),
                                                      'max_abs_diff': float(np.abs(ref[q] - new[q]).max())} for q in ('FP', 'L1')}
        del model
    # (c) G: this code vs stageC_eval.forward_states (padded new data) and vs the saved stage-C Q2 (stage-B set)
    exp, s, e = 'v12_s2_pref_mixed_s0', 0, 200
    model = load_model('G', 'p50', 0, e, dev)
    hook = SC.Hook(model.other_vision_encoder_module)
    for v in ('N', 'O'):
        g = h5py.File(os.path.join(od, f'padded_p80_{v}.h5'), 'r')['test']
        old = SC.forward_states(model, hook, g, n, dev, zero=False)
        new = run(model, D[v]['sv'], D[v]['sm'], None, VMODES['V0'], dev, want_q=True)
        R[f'c|{exp}|ep{e}|{v}'] = {'Q_bit_identical_80': bool(np.array_equal(old['Q'][:, :NSTEP], new['Q'])),
                                   'H_bit_identical_80': bool(np.array_equal(old['H'][:, :NSTEP], new['H'])),
                                   'Q_max_abs_diff': float(np.abs(old['Q'][:, :NSTEP] - new['Q']).max())}
    newb = run(model, svb, smb, None, VMODES['V0'], dev, nstep=100, want_q=True)
    q2 = np.load(os.path.join(RES, 'paper_extra', 'StageC', 'eval', f'{exp}_ep{e}.npz'))['Q2'][:n]
    R[f'c|{exp}|ep{e}|stageB_saved_Q2'] = {'bit_identical': bool(np.array_equal(q2, newb['Q'])),
                                          'max_abs_diff': float(np.abs(q2 - newb['Q']).max())}
    del model
    # (d) timing of one run (600 episodes, 80 steps) per family
    DD = load_data(80)
    R['timing_s'] = {}
    for fam, cond in (('mse', 'p50'), ('G', 'p50')):
        model = load_model(fam, cond, 0, 200, dev)
        t0 = time.time()
        run(model, DD['N']['sv'], DD['N']['sm'], None if fam == 'G' else DD['N']['PT'], VMODES['V1'], dev, want_h1=True)
        R['timing_s'][f'{fam}_one_run_with_h1'] = time.time() - t0
        del model
    keys_a = [k for k in R if k.startswith('a|')]
    R['pass_a_first_half'] = all(R[k][q]['first_half_bit_identical'] for k in keys_a for q in ('FP', 'L1'))
    R['pass_b'] = all(R[k][q]['bit_identical'] for k in R if k.startswith('b|') for q in ('FP', 'L1'))
    R['pass_c'] = all(R[k].get('Q_bit_identical_80', R[k].get('bit_identical')) for k in R if k.startswith('c|'))
    json.dump(R, open(os.path.join(od, 'eval_check.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1), flush=True)


def all_jobs():
    """every checkpoint with its cost in runs; greedy split over workers."""
    jobs = []
    for fam, nrun in (('mse', 48), ('l1', 32)):
        for cond in ('p50', 'none', 'sr50'):
            for s in range(6):
                for e in (160, 170, 180, 190, 200):
                    jobs.append((f'{fam}:{cond}:{s}:{e}', nrun))
    for cond in ('p50', 'none'):
        for s in range(4):
            for e in (160, 170, 180, 190, 200):
                jobs.append((f'G:{cond}:{s}:{e}', 24))
    for cond in ('pre', 'pre_none'):
        for s in range(4):
            jobs.append((f'G:{cond}:{s}:0', 24))
    return jobs


def split_jobs(n_workers):
    load, out = [0] * n_workers, [[] for _ in range(n_workers)]
    for j, c in all_jobs():
        i = int(np.argmin(load))
        out[i].append(j)
        load[i] += c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--jobs', nargs='*', default=[], help='fam:cond:s:e')
    ap.add_argument('--worker', type=int, default=-1, help='with --n_workers: run this worker\'s share of all jobs')
    ap.add_argument('--n_workers', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.check:
        return check(dev)
    od = os.path.join(OUT, 'eval')
    os.makedirs(od, exist_ok=True)
    jobs = args.jobs if args.worker < 0 else split_jobs(args.n_workers)[args.worker]
    DD = {p: load_data(p) for p in PS}
    for job in jobs:
        fam, cond, s, e = job.split(':')
        p = os.path.join(od, f'{fam}_{cond}_s{s}_ep{e}.npz')
        if os.path.exists(p):
            print('exists', p, flush=True)
            continue
        sav, dt = evaluate(fam, cond, int(s), int(e), dev, DD)
        np.savez_compressed(p + '.tmp.npz', **sav)
        os.replace(p + '.tmp.npz', p)
        print(job, f'{dt:.0f}s', len(sav), time.strftime('%m-%d %H:%M'), flush=True)


if __name__ == '__main__':
    main()
