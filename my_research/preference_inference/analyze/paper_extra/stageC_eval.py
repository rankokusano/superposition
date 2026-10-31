"""
Paper round 14 (stage C, G diagnosis), evaluation on the stage-B set (docs/paper_extra_log.md Sec.38.5; pre-registration
3ade5ed). No training; G and the stage-I predictor are fixed. Protocol as the existing stage-2 evaluation
(analyze/v12/s2run.py S2Feed + analyze/v11/eval_s2_v11.run_q2): t = 0 vision shown, t > 0 hidden (both processes),
A-1's true motion fed, model.eval() (no dropout), util.seed_all(0), cudnn deterministic, batch 50. Step t = 0..99
takes frame t and predicts frame t + 1.
Per checkpoint it saves (npz, float32):
  O  other-encoder output before masking (G's input, 64), S  G's LSTMCell h (128), Q  Q2 (8), H  h2 with G's Q2 (128),
  H0 h2 with Q2 = 0 at every step (128): per episode the mean over steps 1-50, the mean over steps 51-99 and step 99;
  Q2 per step (N, 100, 8) (and u per step for the one-bit module);
  gradients of the training loss (self-vision L1 + feature prediction from h1 + from h2, summed over steps, per episode
  the mean over its own elements; states not detached) w.r.t. the process-2 pleasure input Q2_t (t = 0..99), at
  point G (G's output values) and point 0 (zeros), for the total loss and the self-vision term (feature = difference):
  per step (8 dims and the projection on c+/|c+|) for the primary G, per episode g_e (minus the projection summed over
  steps 1-50, 1-25, 26-50) for every G.
Modes:
  --check   Sec.38.5.7 (GPU1 by Kusano-san): v12_s2_pref_mixed_s0 ep200 twice on the same GPU (bit identity of every
            saved array), and against the past: T5 / T2 rebuilt with this code vs the saved per-episode values, and this
            code's Q2 vs the existing code's Q2 (s2run + run_q2) on the same GPU. Size and time measured.
  --models <exp>:<seed>:<epoch>[,<epoch>...] ...  full evaluation (after a separate GPU permission).
-> data/result/paper_extra/StageC/eval/<exp>_ep<e>.npz ; check -> StageC/check/summary.json
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
for p in (os.path.join(_PI, 'analyze', 'v14'), os.path.join(_PI, 'analyze', 'v12'), os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)
import s2run_v14  # noqa  (registers the v14 one-bit class for load_v12; imports s2run)
import s2run  # noqa
import eval_s2_v11 as E  # noqa
from eval_trial_v12 import RES, load_v12  # noqa
import util  # noqa

eval_s2_v11 = E
E.SETS['vdirb'] = ('paper_vdir_b_pref', 'test')
OUT = os.path.join(RES, 'paper_extra', 'StageC')
CP = np.array(json.load(open(os.path.join(RES, 'v14', 'cand_shape.json')))['c_plus'], np.float32)
CPU_ = CP / np.linalg.norm(CP)
PRIMARY = ('v12_s2_pref_mixed', 'v12_s2_pref_random')


def batches(g, N, B):
    for b0 in range(0, N, B):
        b1 = min(b0 + B, N)
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b1])))
        m = torch.from_numpy(g['self_motion'][b0:b1]).float()
        yield b0, b1, x, m


class Hook:
    def __init__(self, module):
        self.out = None
        module.register_forward_hook(self)

    def __call__(self, mod, inp, out):
        self.out = out


def p2_input(q):
    z = torch.zeros(q.shape[0], 1, device=q.device, dtype=q.dtype)
    return torch.cat([z, z, q, torch.ones_like(z)], 1)


def summ(seq):
    """seq (B, 100, d) -> (mean steps 1-50, mean steps 51-99, step 99)."""
    return seq[:, 1:51].mean(1), seq[:, 51:100].mean(1), seq[:, 99]


@torch.no_grad()
def forward_states(model, hook, g, N, dev, zero, B=50):
    util.seed_all(0)
    out = {k: [] for k in ('O', 'S', 'Q', 'H', 'u')}
    for b0, b1, x, m in batches(g, N, B):
        x, m = x.to(dev), m.to(dev)
        model.init_state(b1 - b0)
        seqs = {k: [] for k in out}
        for t in range(100):
            pm = 0 if t == 0 else 1
            p2 = p2_input(torch.zeros(b1 - b0, 8, device=dev)) if zero else None
            o = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=p2)
            seqs['O'].append(hook.out.float().cpu().numpy())
            seqs['S'].append(model.motion_generator_module.state['motion_generator'][0].cpu().numpy())
            seqs['Q'].append(o['q2_hat'].cpu().numpy())
            seqs['H'].append(model.get_state()['other'][0].cpu().numpy())
            if 'u' in o:
                seqs['u'].append(o['u'][:, 0].cpu().numpy())
        for k in out:
            if seqs[k]:
                out[k].append(np.stack(seqs[k], 1))
    return {k: np.concatenate(v) for k, v in out.items() if v}


def gradients(model, g, N, dev, Qvals, B=50):
    """per episode d(loss)/d(Q2_t), t = 0..99, at the given Q2 values (N, 100, 8): total and self-vision term."""
    util.seed_all(0)
    G_tot, G_vis = [], []
    for b0, b1, x, m in batches(g, N, B):
        x, m = x.to(dev), m.to(dev)
        model.init_state(b1 - b0)
        q = torch.from_numpy(Qvals[b0:b1]).to(dev).requires_grad_(True)
        Lvis = torch.zeros(b1 - b0, device=dev)
        Lfp = torch.zeros(b1 - b0, device=dev)
        for t in range(100):
            pm = 0 if t == 0 else 1
            o = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=p2_input(q[:, t]))
            tgt = x[:, t + 1]
            Lvis = Lvis + F.l1_loss(o['self_vision'], tgt, reduction='none').mean((1, 2, 3))
            st = model.get_state()
            sv_enc = model.self_vision_encoder_module(tgt).detach()
            ov_enc = model.other_vision_encoder_module(tgt).detach()
            Lfp = Lfp + F.mse_loss(model.predict_feature(st['self'][0]), sv_enc, reduction='none').mean(1) \
                + F.mse_loss(model.predict_feature(st['other'][0]), ov_enc, reduction='none').mean(1)
        gt, = torch.autograd.grad((Lvis + Lfp).sum(), q, retain_graph=True)
        gv, = torch.autograd.grad(Lvis.sum(), q)
        G_tot.append(gt.cpu().numpy()); G_vis.append(gv.cpu().numpy())
    return np.concatenate(G_tot), np.concatenate(G_vis)


def evaluate(exp, s, e, dev, setname='vdirb', n_ep=0, primary=None):
    model, _ = load_v12(exp, s, e, dev)
    model.eval()
    hook = Hook(model.other_vision_encoder_module)
    d = E.load_set(setname)
    g = h5py.File(d['path'], 'r')[d['split']]
    N = n_ep or g['self_vision'].shape[0]
    t0 = time.time()
    A = forward_states(model, hook, g, N, dev, zero=False)
    Z = forward_states(model, hook, g, N, dev, zero=True)
    sav = {}
    for k in ('O', 'S', 'Q', 'H'):
        for nm, v in zip(('m1_50', 'm51_99', 'last'), summ(A[k])):
            sav[f'{k}_{nm}'] = v.astype(np.float32)
    for nm, v in zip(('m1_50', 'm51_99', 'last'), summ(Z['H'])):
        sav[f'H0_{nm}'] = v.astype(np.float32)
    sav['Q2'] = A['Q'].astype(np.float32)
    if 'u' in A:
        sav['u'] = A['u'].astype(np.float32)
    t_fwd = time.time() - t0
    primary = (exp.rsplit('_s', 1)[0] in PRIMARY) if primary is None else primary
    for pt, Qv in (('G', A['Q'].astype(np.float32)), ('0', np.zeros_like(A['Q'], np.float32))):
        gt, gv = gradients(model, g, N, dev, Qv)
        for lab, gr in (('tot', gt), ('vis', gv)):
            proj = -(gr @ CPU_)                                         # (N, 100), minus gradient on c+
            sav[f'ge_{lab}_{pt}_1_50'] = proj[:, 1:51].sum(1).astype(np.float32)
            sav[f'ge_{lab}_{pt}_1_25'] = proj[:, 1:26].sum(1).astype(np.float32)
            sav[f'ge_{lab}_{pt}_26_50'] = proj[:, 26:51].sum(1).astype(np.float32)
            if primary:
                sav[f'grad_{lab}_{pt}'] = gr.astype(np.float32)
                sav[f'gproj_{lab}_{pt}'] = proj.astype(np.float32)
    return sav, {'t_forward_s': t_fwd, 't_total_s': time.time() - t0, 'N': N}


# ------------------------------------------------------------------ check (Sec.38.5.7)
def old_q2(exp, s, e, setname, dev):
    m, _ = load_v12(exp, s, e, dev)
    f = s2run.S2Feed(m)
    d = E.load_set(setname)
    f.reset(h5py.File(d['path'], 'r')[d['split']]['self_motion'][()])
    Q, _ = E.run_q2(f, d['path'], d['split'], dev)
    return Q, d


def new_q2(exp, s, e, setname, dev):
    model, _ = load_v12(exp, s, e, dev)
    model.eval()
    hook = Hook(model.other_vision_encoder_module)
    d = E.load_set(setname)
    g = h5py.File(d['path'], 'r')[d['split']]
    return forward_states(model, hook, g, g['self_vision'].shape[0], dev, zero=False)['Q'], d


def t2_t5(Qg, Qs, Qr, ds):
    def gsel(q, idx):
        return np.take_along_axis(q, idx, 1).mean(1)
    L0 = ds['layout']
    newG = np.array([E.RB[int(l)] for l in L0]); oldG = np.array([E.GB[int(l)] for l in L0])
    sT2 = (gsel(Qs[:, 50], newG) - gsel(Qs[:, 50], oldG)) - (gsel(Qg[:, 50], newG) - gsel(Qg[:, 50], oldG))
    pairs = np.load(os.path.join(RES, 'v12', 'stage2', 'eval', 't5_pairs.npz'))['pairs']
    Lp = pairs[:, 0]

    def gr(q, L):
        return gsel(q, np.array([E.GB[int(l)] for l in L])) - gsel(q, np.array([E.RB[int(l)] for l in L]))
    dT5 = gr(Qg[pairs[:, 1], pairs[:, 2]], Lp) - gr(Qr[pairs[:, 3], pairs[:, 4]], Lp)
    return sT2, dT5


def check(dev):
    exp, s, e = 'v12_s2_pref_mixed_s0', 0, 200
    od = os.path.join(OUT, 'check')
    os.makedirs(od, exist_ok=True)
    R = {'checkpoint': f'{exp} ep{e}', 'device': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}
    # (1) same GPU twice
    runs = []
    for i in range(2):
        sav, tm = evaluate(exp, s, e, dev, primary=True)
        p = os.path.join(od, f'{exp}_ep{e}_run{i + 1}.npz')
        np.savez_compressed(p, **sav)
        runs.append(sav)
        R[f'run{i + 1}'] = {**tm, 'file_bytes': os.path.getsize(p)}
        print('run', i + 1, R[f'run{i + 1}'], flush=True)
    R['same_gpu_bit_identical'] = {k: bool(np.array_equal(runs[0][k], runs[1][k])) for k in runs[0]}
    R['same_gpu_all'] = all(R['same_gpu_bit_identical'].values())
    R['n_arrays'] = len(runs[0])
    # (2) past values: new code vs existing code (same GPU) and vs the saved per-episode T2 / T5
    t0 = time.time()
    Qn, Qo, D = {}, {}, {}
    for st in ('green', 'swap', 'red'):
        Qn[st], D[st] = new_q2(exp, s, e, st, dev)
        Qo[st], _ = old_q2(exp, s, e, st, dev)
    R['q2_new_vs_old_code'] = {st: {'bit_identical': bool(np.array_equal(Qn[st], Qo[st])),
                                    'max_abs_diff': float(np.abs(Qn[st].astype(np.float64) - Qo[st]).max())} for st in Qn}
    sv = np.load(os.path.join(RES, 'v12', 'stage2', 'eval', 'pref_mixed_s0_ep200_per.npz'))
    for nm, Q in (('new', Qn), ('old', Qo)):
        sT2, dT5 = t2_t5(Q['green'], Q['swap'], Q['red'], D['swap'])
        R[f'past_{nm}_code'] = {
            'T2_bit_identical': bool(np.array_equal(sT2.astype(sv['sT2_t50'].dtype), sv['sT2_t50'])),
            'T2_max_abs_diff_per_episode': float(np.abs(sT2 - sv['sT2_t50']).max()),
            'T2_mean': float(sT2.mean()), 'T2_mean_saved': float(sv['sT2_t50'].mean()),
            'T5_bit_identical': bool(np.array_equal(dT5.astype(sv['dT5'].dtype), sv['dT5'])),
            'T5_max_abs_diff_per_pair': float(np.abs(dT5 - sv['dT5']).max()),
            'T5_mean': float(dT5.mean()), 'T5_mean_saved': float(sv['dT5'].mean())}
        r = R[f'past_{nm}_code']
        r['T2_mean_diff'] = abs(r['T2_mean'] - r['T2_mean_saved'])
        r['T5_mean_diff'] = abs(r['T5_mean'] - r['T5_mean_saved'])
        r['sign_and_pass_form_unchanged'] = bool(np.sign(r['T2_mean']) == np.sign(r['T2_mean_saved']) and
                                                 np.sign(r['T5_mean']) == np.sign(r['T5_mean_saved']) and
                                                 (r['T2_mean'] > 0) == (r['T2_mean_saved'] > 0) and
                                                 (r['T5_mean'] > 0) == (r['T5_mean_saved'] > 0))
    R['t_past_check_s'] = time.time() - t0
    json.dump(R, open(os.path.join(od, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != 'same_gpu_bit_identical'}, indent=0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--models', nargs='*', default=[])
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.check:
        return check(dev)
    os.makedirs(os.path.join(OUT, 'eval'), exist_ok=True)
    for spec in args.models:
        exp, s, eps = spec.split(':')
        for e in map(int, eps.split(',')):
            p = os.path.join(OUT, 'eval', f'{exp}_ep{e}.npz')
            sav, tm = evaluate(exp, int(s), e, dev)
            np.savez_compressed(p, **sav)
            print(exp, e, tm, os.path.getsize(p), flush=True)


if __name__ == '__main__':
    main()
