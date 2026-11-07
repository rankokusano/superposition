"""
Paper round 26 (paper/check/paper_instructions_round26.md; docs/paper_extra_log.md Sec.50; exploratory, outside the
pre-registration). Why the other's preference-tied prediction is not learned: the learning signal (Sec.1), the upper
bound of the entrance (Sec.2), the collapse of the self rule (Sec.3). GPU (GPU1-3) except data31. New code; existing
models / data / results are only read. Outputs: data/result/paper_extra/Round26/.

Terms (instruction, this round only)
  T  pleasure from A-2's view (other_vision, same frame as A-1's view) with A-2's OWN liking:
     T = scale * pleasure_torch(view, Red, Green) * (+1 red-liker / -1 green-liker); ranges 2-5 set to 0.
  W  the same with the opposite liking (= -T exactly, since pleasure(liked, disliked) is antisymmetric).
  Z  zeros.
  The other pathway's input is process-2 = [0, 0, X, 1] (X in {T, W, Z}), the stage-II form.
  Flow V = the stage-II training flow as is (MotionGenerationPredictionFeaturePredictionRunner: the LSTM's vision is
     masked at every t > 0; the current view enters every step only through process-1's pleasure P1 (A-1's current
     view, motion_mask off in stage II) and through the other pathway's input (here X from A-2's current view)).
  Flow B = V, but from frame s+3 for 15 steps (t = s+3 .. s+17) the view-derived inputs are stopped: P1 and X are held
     at their values of frame s+2 (frames s, s+1, s+2 after the swap are "shown", i.e. current); A-1's motion is fed
     (hidden flag 0); the LSTM's vision stays masked as in V. s = swap frame (first frame of the new layout); in an
     episode without a swap s = a pseudo frame drawn uniform 20-80 once per episode (default_rng(26), fixed).
     (Round 21's way of stopping the vision: hold the last shown view's inputs. In the training flow the LSTM never
     sees the view at t > 0, so what B stops is the view-derived pleasure inputs.)
  Window = steps t = s .. s+19 (inputs frames s..s+19, targets frames s+1..s+20).

Modes
  check    flow / loss reproduces the runner; T definition; B holds; R25Model(stage-I mode) == stage-I model; timings.
  signal   Sec.1: stage-I predictors (none / p50 / sr50 / p80, pairs 1-6 = seeds 0-5, l1 ep200), no adapter; per episode
           the window sums of the stage-II loss components [self_vision L1, feature_prediction_self,
           feature_prediction_other] for X in {T, W, Z} x flow {V, B}; data: paper_r25_swap50 train (3000) and round 20
           p_rule 0.8 S (no swap) / O (swap), s = 50 (600 each). -> Round26/signal/<cond>_s<pair>.npz
  train2   Sec.2: R25Model (adapter Linear(8,8) identity, ranges 2-5 masked after it = M) on the stage-I predictor of the
           pair, G replaced by T (p2 = [0, 0, mask(A(T)), 1]); only the adapter is trained; stage-II loss; AdamW lr 1e-3
           wd 0.01 (bias 0), batch 10, 50 epochs, data paper_r25_swap50 train in flow V or B (B in every episode).
           Evaluation every 5 epochs incl. 0: round 24 Sec.1.5 test (round 20 p_rule 0.8, S / O, frames 0..52 shown
           with T through the adapter, 53..62 frozen) with input T (T of frame 52 held, through the adapter and mask)
           or Z (zeros fed unchanged); Delta per episode; adapter weights. -> Round26/runs/s2_<cond>_<V|B>_s<pair>/
  data31   Sec.3.1 (CPU): stage-I data v11_s1_mixed, stage-II data v11_s2_pref, round-25 data; A-1's moves vs its own
           pleasure input. -> Round26/data31/summary.json
  train32  Sec.3.2: round 25 stage 2 (p50, pair 1, from Round25 stage1_p50_M ep100, adapter + G + shared LSTM, LSTM lr
           ratio 0.01 / 0.1, 50 epochs, round-25 evaluations) but every batch = 10 stage-II episodes (round-25 flow and
           loss) + 10 stage-I episodes (v11_s1_mixed train; stage-I flow: PredictionFeaturePredictionRunner, vision mask
           0.99 for t > 0, motion-masking schedule on, process-2 = zeros(11); stage-I loss of the l1 period,
           v12_s1_mixed_l1_s0.yml: l1); loss = mean over the 20 = 0.5 (L_II + L_I). -> Round26/runs/s32_p50_r<ratio>/
Usage: python analyze/paper_extra/round26.py check | signal --jobs p50:0 ... | train2 --cond p50 --mode V --pair 0 |
       data31 | train32 --ratio 0.1
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round25_train as R25  # noqa
import round21_eval as R21  # noqa
import round24_eval as R24E  # noqa
from p2test_v11 import read_a2  # noqa
import util  # noqa
from model import model_v12  # noqa
from exp import runner as runners, loader  # noqa
from v11core.pleasure import pleasure_torch  # noqa

RES, _PI = R25.RES, R25._PI
OUT = os.path.join(RES, 'paper_extra', 'Round26')
D25 = R25.DATA
DS1 = os.path.join(_PI, 'data', 'data', 'v11_s1_mixed', 'data.h5')
DS2 = os.path.join(_PI, 'data', 'data', 'v11_s2_pref', 'data.h5')
S1EXP = {'none': 'v12_s1_random_l1_s{s}', 'p50': 'v12_s1_mixed_l1_s{s}',
         'sr50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s{s}', 'p80': 'v13_s1_mixed_l1_s{s}'}
WIN, SHOW, FRZ = 20, 3, 15
PSEUDO_SEED = 26
INPUTS, FLOWS = ('T', 'W', 'Z'), ('V', 'B')
SIGN = {'T': 1.0, 'W': -1.0, 'Z': 0.0}
KEYS = ('self_vision', 'other_vision', 'self_motion', 'self_position', 'other_motion', 'other_position')


# ---------------------------------------------------------------- common
def freeze_frames(sw):
    ps = np.random.default_rng(PSEUDO_SEED).integers(20, 81, len(sw))
    return np.where(sw >= 0, sw, ps).astype(np.int64)


def load25(split='train', n=0):
    g = h5py.File(D25, 'r')[split]
    sw = g['swap_frame'][()].astype(np.int64)
    s = freeze_frames(sw)
    sl = slice(0, n or None)
    d = {k: g[k][sl] for k in KEYS}
    return d, sw[sl], s[sl], g['a2_liked'][sl].astype(np.int64)


def sign_of(lk, dev):
    return torch.from_numpy(np.where(np.asarray(lk) == 1, 1.0, -1.0).astype(np.float32)).to(dev)


def t_pleasure(m, ov, sign):
    """ov (B, 3, 16, 64) scaled to [-1, 1]; sign (B,) +1 red-liker, -1 green-liker -> T (B, 8), ranges 2-5 = 0."""
    P = m.pleasure_scale * pleasure_torch((ov + 1) / 2, 'Red', 'Green') * sign[:, None]
    P[:, 2:6] = 0.0
    return P


class Holder:
    """Flow B: per episode, P1 and X are held at frame s+2 for t = s+3 .. s+17."""

    def __init__(self, s):
        self.s, self.p1, self.x = s, None, None

    def __call__(self, t, P1, X):
        hold_now = (t == self.s + SHOW - 1)
        frz = (t >= self.s + SHOW) & (t < self.s + SHOW + FRZ)
        if self.p1 is None:
            self.p1, self.x = P1.clone(), X.clone()
        self.p1 = torch.where(hold_now[:, None], P1, self.p1)
        self.x = torch.where(hold_now[:, None], X, self.x)
        return torch.where(frz[:, None], self.p1, P1), torch.where(frz[:, None], self.x, X), frz


def p2_of(X):
    B = X.shape[0]
    return torch.cat([torch.zeros(B, 2, device=X.device), X, torch.ones(B, 1, device=X.device)], 1)


def s1_path(cond, pair):
    return os.path.join(RES, S1EXP[cond].format(s=pair), str(pair), 'model', '00200.pth')


def build_plain(cond, pair, dev):
    """Stage-II model class with the stage-I predictor (G untrained, never used: process-2 always overridden)."""
    _, mcfg, _ = R25.configs()
    util.seed_all(0)
    m = model_v12.SuperpositionNetworkMotionPleasureHoldOther(mcfg)
    res = m.load_state_dict(torch.load(s1_path(cond, pair), map_location='cpu')['model'], strict=False)
    assert not res.unexpected_keys and all(k.startswith('motion_generator_module') for k in res.missing_keys)
    return m.to(dev).eval()


def build_r25(cond, pair, dev):
    """R25Model (variant M) on the stage-I predictor of the pair (seed_all(0) before construction, as round 25)."""
    _, mcfg, _ = R25.configs()
    util.seed_all(0)
    m = R25.R25Model(mcfg, True)
    res = m.load_state_dict(torch.load(s1_path(cond, pair), map_location='cpu')['model'], strict=False)
    assert not res.unexpected_keys
    assert all(k.startswith(('motion_generator_module', 'p2_adapter', 'p2_mask')) for k in res.missing_keys)
    return m.to(dev)


# ---------------------------------------------------------------- Sec.1
@torch.no_grad()
def window_losses(m, x, o, mo, sign, s, flow, inp, all_steps=False):
    """x, o: scaled (B, T, 3, 16, 64); mo (B, T, 2); s (B,) long. -> (B, 3) sums over the window (or all steps) of
    [self_vision L1, feature_prediction_self, feature_prediction_other] per episode (as the runner, per episode).
    inp 'G' = process-2 not overridden (the model's own G; used by the check only)."""
    B, Tn = x.shape[0], x.shape[1]
    tmax = Tn - 1 if all_steps else int(s.max()) + WIN
    m.init_state(B)
    hold = Holder(s)
    out = torch.zeros(B, 3, device=x.device)
    for t in range(tmax):
        xt = {'self_vision': x[:, t], 'self_motion': mo[:, t]}
        p1 = p2 = None
        if inp != 'G':
            X = t_pleasure(m, o[:, t], sign) * SIGN[inp]
            if flow == 'B':
                p1, X, _ = hold(t, m.compute_pleasure(x[:, t]), X)
            p2 = p2_of(X)
        p = 0 if t == 0 else 1
        pred = m(xt, p, p, p1_override=p1, p2_override=p2)
        inw = torch.ones(B, dtype=torch.bool, device=x.device) if all_steps else (t >= s) & (t < s + WIN)
        if inw.any():
            tr = x[:, t + 1]
            l1 = (pred['self_vision'] - tr).abs().flatten(1).mean(1)
            st = m.get_state()
            fs = ((m.predict_feature(st['self'][0]) - m.self_vision_encoder_module(tr)) ** 2).flatten(1).mean(1)
            fo = ((m.predict_feature(st['other'][0]) - m.other_vision_encoder_module(tr)) ** 2).flatten(1).mean(1)
            out += torch.stack([l1, fs, fo], 1) * inw[:, None].float()
    return out.cpu().numpy()


def sig_dataset(m, D, s, lk, dev, batch=100):
    res = {f'{f}_{i}': [] for f in FLOWS for i in INPUTS}
    N = D['self_vision'].shape[0]
    for b0 in range(0, N, batch):
        sl = slice(b0, b0 + batch)
        sb = torch.from_numpy(s[sl]).to(dev)
        tn = int(s[sl].max()) + WIN + 1
        x = R21.scaled(D['self_vision'][sl, :tn], dev)
        o = R21.scaled(D['other_vision'][sl, :tn], dev)
        mo = torch.from_numpy(D['self_motion'][sl, :tn]).float().to(dev)
        sg = sign_of(lk[sl], dev)
        for f in FLOWS:
            for i in INPUTS:
                res[f'{f}_{i}'].append(window_losses(m, x, o, mo, sg, sb, f, i))
    return {k: np.concatenate(v).astype(np.float32) for k, v in res.items()}


def signal(jobs):
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    od = os.path.join(OUT, 'signal')
    os.makedirs(od, exist_ok=True)
    d25, sw, s25, lk25 = load25('train')
    r20 = {v: R24E.load(v) for v in ('S', 'O')}
    for job in jobs:
        cond, pair = job.split(':')
        pair = int(pair)
        path = os.path.join(od, f'{cond}_s{pair}.npz')
        if os.path.exists(path):
            print('exists', path)
            continue
        t0 = time.time()
        m = build_plain(cond, pair, dev)
        sav = {'r25_swap': (sw >= 0), 'r25_s': s25, 'r25_liked': lk25}
        for k, v in sig_dataset(m, d25, s25, lk25, dev).items():
            sav[f'r25_{k}'] = v
        for var, D in r20.items():
            s = np.full(D['self_vision'].shape[0], 50, np.int64)
            for k, v in sig_dataset(m, D, s, D['a2_liked'], dev).items():
                sav[f'r20{var}_{k}'] = v
            sav[f'r20{var}_liked'] = D['a2_liked']
        sav['sec'] = np.float32(time.time() - t0)
        np.savez_compressed(path, **sav)
        print(json.dumps({'job': job, 'sec': round(time.time() - t0, 1)}), flush=True)


# ---------------------------------------------------------------- Sec.2
class Eval2:
    def __init__(self, dev, n=0):
        self.dev = dev
        self.ab = R21.calib_common()
        self.r20 = {v: R24E.load(v) for v in ('S', 'O')}
        if n:
            self.r20 = {v: {k: a[:n] for k, a in D.items()} for v, D in self.r20.items()}
        self.lo, self.ln = R24E.targets(self.r20['S']['layout'], self.r20['S']['a2_liked'])

    @torch.no_grad()
    def other(self, m, batch=50):
        """Round 24 Sec.1.5 test with T: frames 0..52 shown with p2 = [0, 0, mask(A(T_t)), 1]; frames 53..62 frozen
        (vision masked, motion hidden, p1 = 0) with T of frame 52 held through A and the mask ('T') or zeros fed
        unchanged ('Z')."""
        m.eval()
        T0, K = R24E.T0, R24E.K
        out = {}
        for v, D in self.r20.items():
            N = D['self_vision'].shape[0]
            a2 = D['other_position'][:, T0 - 1].astype(np.float64)
            sp = D['self_position'][:, T0 - 1:T0 - 1 + K].astype(np.float64)
            preds = {'T': [], 'Z': []}
            traw, tin = [], []
            for b0 in range(0, N, batch):
                x = R21.scaled(D['self_vision'][b0:b0 + batch, :T0 + K], self.dev)
                o = R21.scaled(D['other_vision'][b0:b0 + batch, :T0], self.dev)
                mo = torch.from_numpy(D['self_motion'][b0:b0 + batch]).float().to(self.dev)
                sg = sign_of(D['a2_liked'][b0:b0 + batch], self.dev)
                B = x.shape[0]
                for inp in ('T', 'Z'):
                    util.seed_all(0)
                    m.init_state(B)
                    m.p2_bypass = False
                    for t in range(T0):
                        X = t_pleasure(m, o[:, t], sg)
                        m({'self_vision': x[:, t], 'self_motion': mo[:, t]}, 0, 0, p2_override=p2_of(X))
                    Q = torch.zeros(B, 11, device=self.dev)
                    Q[:, 10] = 1.0
                    if inp == 'T':
                        Q[:, 2:10] = X
                        if v == 'S':
                            traw.append(X.abs().cpu().numpy())
                            tin.append(m.p2_pleasure(X).abs().cpu().numpy())
                    else:
                        m.p2_bypass = True
                    P1 = torch.zeros(B, 8, device=self.dev)
                    hid = torch.ones(B, dtype=torch.bool, device=self.dev)
                    pr = []
                    for k in range(K):
                        r = m({'self_vision': x[:, T0 + k], 'self_motion': mo[:, T0 + k]}, 1.0, 1.0, p1_override=P1,
                              p2_override=Q, hidden_override=hid)
                        pr.append(((r['self_vision'] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
                    m.p2_bypass = False
                    preds[inp].append(np.stack(pr, 1))
            for inp, pl in preds.items():
                pr = np.concatenate(pl)
                est = np.stack([read_a2(pr[:, i], sp[:, i], *self.ab) for i in range(K)], 1)
                S = R21.other_slopes(est, a2, self.lo, self.ln)
                out[f'{v}_{inp}_delta'] = (S[:, 0] - S[:, 1]).astype(np.float32)
                out[f'{v}_{inp}_readable'] = np.float32(np.isfinite(est[..., 0]).mean())
            if v == 'S':
                out['t_abs_raw'] = np.concatenate(traw).astype(np.float32)
                out['t_abs_in'] = np.concatenate(tin).astype(np.float32)
        out['a2_liked'] = self.r20['S']['a2_liked']
        out['adapter_w'] = m.p2_adapter.weight.detach().cpu().numpy()
        out['adapter_b'] = m.p2_adapter.bias.detach().cpu().numpy()
        return out


def seq_T(m, seq, ep, cfg, runner, flow, s, sign):
    """One batch in the stage-II training flow with process-2 = [0, 0, mask(A(T)), 1] (flow V) or with the B holds."""
    loss = defaultdict(lambda: 0)
    hold = Holder(s)
    for x, truth, t in seq:
        if t == 0:
            m.init_state(x['self_vision'].shape[0])
        X = t_pleasure(m, x['other_vision'], sign)
        p1 = None
        if flow == 'B':
            p1, X, _ = hold(t, m.compute_pleasure(x['self_vision']), X)
        p = runner.gen_mask_prob(t, cfg)
        pred = m(x, p, p, p1_override=p1, p2_override=p2_of(X))
        runner.calc_loss(m, loss, pred, truth, ep, t, cfg)
    return loss


def train2(cond, flow, pair, n_ep=50, every=5):
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    name = f's2_{cond}_{flow}_s{pair}'
    od = os.path.join(OUT, 'runs', name)
    assert not os.path.exists(os.path.join(od, 'done.json')), od
    for sub in ('model', 'eval'):
        os.makedirs(os.path.join(od, sub), exist_ok=True)
    exp_cfg, _, _ = R25.configs()
    m = build_r25(cond, pair, dev)
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith('p2_adapter')
    opt = R25.optimizer(m, 1.0)
    d, sw, s, lk = load25('train')
    tl = loader.DataLoader(d, exp_cfg.train.batch_size, True, dev)
    ev = Eval2(dev)
    runner = runners.MotionGenerationPredictionFeaturePredictionRunner()
    log = {'name': name, 'cond': cond, 'flow': flow, 'pair': pair, 'start': s1_path(cond, pair), 'epochs': []}

    def evaluate(ep):
        t0 = time.time()
        np.savez_compressed(os.path.join(od, 'eval', f'ep{ep:03d}.npz'), **ev.other(m))
        torch.save({k: v for k, v in m.state_dict().items() if k.startswith('p2_adapter')},
                   os.path.join(od, 'model', f'{ep:03d}.pth'))
        m.train()
        return time.time() - t0

    log['eval0_sec'] = evaluate(0)
    util.seed_all(R25.SEED)
    m.train()
    for ep in range(1, n_ep + 1):
        t0 = time.time()
        acc, nb = defaultdict(float), 0
        for seq, bidx in tl.load_sequence():
            loss = seq_T(m, seq, ep, exp_cfg, runner, flow, torch.from_numpy(s[bidx]).to(dev), sign_of(lk[bidx], dev))
            opt.zero_grad()
            sum(loss.values()).backward()
            opt.step()
            for k, v in loss.items():
                acc[k] += float(v)
            nb += 1
        rec = {'epoch': ep, 'sec': time.time() - t0, **{k: v / nb for k, v in acc.items()}}
        if ep % every == 0:
            rec['eval_sec'] = evaluate(ep)
        log['epochs'].append(rec)
        json.dump(log, open(os.path.join(od, 'log.json'), 'w'), indent=1)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    json.dump({'finished': time.strftime('%Y-%m-%d %H:%M')}, open(os.path.join(od, 'done.json'), 'w'))


# ---------------------------------------------------------------- Sec.3.2
def s1_cfg():
    return util.load_config(os.path.join(_PI, 'config', 'exp', 'v12_s1_mixed_l1_s0.yml'))


def load_s1(split='train', n=0):
    g = h5py.File(DS1, 'r')[split]
    sl = slice(0, n or None)
    return {k: g[k][sl] for k in ('self_vision', 'self_motion', 'self_position', 'other_motion', 'other_position')}


def seq_stage1(m, seq, ep, cfg, runner):
    """One batch in the stage-I flow and loss: motion-masking schedule on (Hold), process-2 = zeros(11) fed unchanged,
    vision mask 0 at t = 0 and cfg.p_mask_vision (0.99) after, PredictionFeaturePredictionRunner loss (l1)."""
    loss = defaultdict(lambda: 0)
    save = m.motion_masking
    m.motion_masking, m.p2_bypass = True, True
    try:
        for x, truth, t in seq:
            if t == 0:
                B = x['self_vision'].shape[0]
                m.init_state(B)
                Z = torch.zeros(B, 11, device=x['self_vision'].device)
            p = runner.gen_mask_prob(t, cfg)
            pred = m(x, p, p, p2_override=Z)
            runner.calc_loss(m, loss, pred, truth, ep, t, cfg)
    finally:
        m.motion_masking, m.p2_bypass = save, False
    return loss


def train32(ratio, n_ep=50, every=5):
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    name = f's32_p50_r{ratio:g}'
    od = os.path.join(OUT, 'runs', name)
    assert not os.path.exists(os.path.join(od, 'done.json')), od
    for sub in ('model', 'eval'):
        os.makedirs(os.path.join(od, sub), exist_ok=True)
    m, exp_cfg = R25.build('p50', True, dev)
    m.load_state_dict(torch.load(os.path.join(R25.OUT, 'runs', 'stage1_p50_M', 'model', '100.pth'), map_location='cpu'))
    R25.set_trainable(m, 2)
    opt = R25.optimizer(m, ratio)
    cfg1 = s1_cfg()
    tl2 = loader.DataLoader(R25.load_split('train'), exp_cfg.train.batch_size, True, dev)
    tl1 = loader.DataLoader(load_s1('train'), cfg1.train.batch_size, True, dev)
    ev = R25.Evaluator(dev)
    run2 = runners.MotionGenerationPredictionFeaturePredictionRunner()
    run1 = runners.PredictionFeaturePredictionRunner()
    log = {'name': name, 'ratio': ratio, 'start': 'Round25/runs/stage1_p50_M/model/100.pth', 'epochs': []}
    torch.save(m.state_dict(), os.path.join(od, 'model', '000.pth'))
    log['eval0_sec'] = R25.evaluate(m, ev, od, 0, True)
    util.seed_all(R25.SEED)
    m.train()
    for ep in range(1, n_ep + 1):
        t0 = time.time()
        acc, nb = defaultdict(float), 0
        for (seq2, _), (seq1, _) in zip(tl2.load_sequence(), tl1.load_sequence()):
            l2, _ = run2.run_seq(m, seq2, ep, exp_cfg)
            l1 = seq_stage1(m, seq1, ep, cfg1, run1)
            opt.zero_grad()
            (0.5 * (sum(l2.values()) + sum(l1.values()))).backward()
            opt.step()
            for k, v in l2.items():
                acc[k] += float(v)
            for k, v in l1.items():
                acc['s1_' + k] += float(v)
            nb += 1
        rec = {'epoch': ep, 'sec': time.time() - t0, **{k: v / nb for k, v in acc.items()}}
        if ep % every == 0:
            torch.save(m.state_dict(), os.path.join(od, 'model', f'{ep:03d}.pth'))
            rec['eval_sec'] = R25.evaluate(m, ev, od, ep, True)
        log['epochs'].append(rec)
        json.dump(log, open(os.path.join(od, 'log.json'), 'w'), indent=1)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    json.dump({'finished': time.strftime('%Y-%m-%d %H:%M')}, open(os.path.join(od, 'done.json'), 'w'))


# ---------------------------------------------------------------- Sec.3.1
def data31():
    """A-1's moves vs its own pleasure input (numpy; per step t = 0..99: frame t's view, motion t = the move from
    frame t's position). Red's place from the layout of frame t. Pleasure P = pleasure_np(A-1's view, Red, Green)
    (the stage-I/II process-1 input, scale 1). Direction of P: v = sum_k P_k u_k (u_k = BIN_UNIT, world-fixed)."""
    from v11core.pleasure import pleasure_np, BIN_UNIT
    from sim_v10 import layout as L
    out = {}
    rng = np.random.default_rng(0)
    for name, path in (('s1_mixed', DS1), ('s2_pref', DS2), ('r25_swap50', D25)):
        f = h5py.File(path, 'r')
        g = f['train']
        N, T = g['self_motion'].shape[:2]
        lay_t = g['layout_t'][()] if 'layout_t' in g else np.repeat(g['layout'][()][:, None], T, 1)
        src = g['a1_source'][()]
        pos = g['self_position'][()].astype(np.float64)
        mot = g['self_motion'][()].astype(np.float64)
        red = {c: np.array(L.LANDMARKS[L.CODE_LAYOUT[int(c)]]['Red'], np.float64) for c in np.unique(lay_t)}
        R = np.zeros(lay_t.shape + (2,))
        for c, p in red.items():
            R[lay_t == c] = p
        per = defaultdict(list)
        for b0 in range(0, N, 250):
            sv = g['self_vision'][b0:b0 + 250, :T - 1]
            P = pleasure_np(sv.astype(np.float32) / 255.0, 'Red', 'Green')           # (b, T-1, 8)
            v = P @ BIN_UNIT                                                            # (b, T-1, 2)
            mo = mot[b0:b0 + 250, :T - 1]
            p0, p1 = pos[b0:b0 + 250, :T - 1], pos[b0:b0 + 250, 1:]
            Rt = R[b0:b0 + 250, :T - 1]
            moving = np.linalg.norm(mo, axis=-1) > 1e-9
            closer = np.linalg.norm(p1 - Rt, axis=-1) < np.linalg.norm(p0 - Rt, axis=-1) - 1e-9
            pnz = np.abs(P).sum(-1) > 0
            vnz = np.linalg.norm(v, axis=-1) > 1e-9
            agree = (v * mo).sum(-1) > 0
            s_ = src[b0:b0 + 250, :T - 1]
            for e in range(sv.shape[0]):
                mv = moving[e]
                per['n_steps'].append(mv.size)
                per['n_moving'].append(int(mv.sum()))
                per['closer_red_moving'].append(int((closer[e] & mv).sum()))
                per['p_nonzero'].append(int(pnz[e].sum()))
                sel = mv & vnz[e]
                per['n_dir'].append(int(sel.sum()))
                per['agree'].append(int((agree[e] & sel).sum()))
                for code, lab in ((1, 'rule'), (0, 'random')):
                    ss = sel & (s_[e] == code)
                    per[f'n_dir_{lab}'].append(int(ss.sum()))
                    per[f'agree_{lab}'].append(int((agree[e] & ss).sum()))
                per['n_rule_steps'].append(int((s_[e] == 1).sum()))
        A = {k: np.array(v, np.float64) for k, v in per.items()}

        def frac(num, den):
            r = A[num].sum() / A[den].sum() if A[den].sum() else float('nan')
            idx = rng.integers(0, len(A[den]), (1000, len(A[den])))
            bs = A[num][idx].sum(1) / np.maximum(A[den][idx].sum(1), 1)
            return {'value': float(r), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    'num': float(A[num].sum()), 'den': float(A[den].sum())}
        out[name] = {'attrs': {k: (v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, 'item') else str(v)))
                               for k, v in f.attrs.items()},
                     'episodes': int(N), 'steps_per_episode': int(T - 1),
                     'moving_share': frac('n_moving', 'n_steps'),
                     'closer_to_red_among_moving': frac('closer_red_moving', 'n_moving'),
                     'pleasure_nonzero_share': frac('p_nonzero', 'n_steps'),
                     'agree_among_moving_with_direction': frac('agree', 'n_dir'),
                     'agree_rule_steps': frac('agree_rule', 'n_dir_rule'),
                     'agree_random_steps': frac('agree_random', 'n_dir_random'),
                     'rule_step_share': frac('n_rule_steps', 'n_steps')}
        print(name, json.dumps({k: (v['value'] if isinstance(v, dict) and 'value' in v else v)
                                for k, v in out[name].items() if k != 'attrs'}), flush=True)
    od = os.path.join(OUT, 'data31')
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, 'summary.json'), 'w'), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- check
def check():
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    out = {}
    exp_cfg, mcfg, s1model = R25.configs()
    # (a) window_losses with all steps and the model's own G == runner.run_seq (round-25 stage1_p50_M ep100, 10 eps)
    m = build_r25('p50', 0, dev)
    m.load_state_dict(torch.load(os.path.join(R25.OUT, 'runs', 'stage1_p50_M', 'model', '100.pth'), map_location='cpu'))
    m.eval()
    d, sw, s, lk = load25('test', 10)
    x = R21.scaled(d['self_vision'], dev)
    o = R21.scaled(d['other_vision'], dev)
    mo = torch.from_numpy(d['self_motion']).float().to(dev)
    mine = window_losses(m, x, o, mo, sign_of(lk, dev), torch.from_numpy(s).to(dev), 'V', 'G', all_steps=True).mean(0)
    runner = runners.MotionGenerationPredictionFeaturePredictionRunner()
    tl = loader.DataLoader({k: d[k] for k in KEYS if k != 'other_vision'}, 10, False, dev)
    with torch.no_grad():
        for seq, _ in tl.load_sequence():
            ref, _ = runner.run_seq(m, seq, 1, exp_cfg)
    ref = np.array([float(ref['self_vision']), float(ref['feature_prediction_self']), float(ref['feature_prediction_other'])])
    out['a_window_losses_vs_runner'] = {'mine': mine.tolist(), 'runner': ref.tolist(),
                                        'max_rel_diff': float(np.max(np.abs(mine - ref) / np.abs(ref)))}
    # (b) T: red-liker == model.compute_pleasure(A-2 view) with bins 2-5 zeroed; green-liker == minus; W == -T;
    #     share of nonzero pleasure in bins 2-5 of A-2's view (should be 0: the colour boxes are on the left wall)
    ov = o[:, :].reshape(-1, 3, 16, 64)
    sg = sign_of(np.repeat(lk, o.shape[1]), dev)
    T = t_pleasure(m, ov, sg)
    raw = m.compute_pleasure(ov)
    out['b_T_red_eq_compute_pleasure'] = bool(torch.equal(T[sg > 0][:, [0, 1, 6, 7]], raw[sg > 0][:, [0, 1, 6, 7]]))
    out['b_T_green_eq_minus'] = bool(torch.equal(T[sg < 0][:, [0, 1, 6, 7]], -raw[sg < 0][:, [0, 1, 6, 7]]))
    out['b_raw_bins2to5_nonzero_share'] = float((raw[:, 2:6].abs() > 0).float().mean())
    out['b_T_nonzero_share'] = float((T.abs().sum(1) > 0).float().mean())
    # (c) B holds: P1 and X equal frame s+2's values on s+3..s+17, current elsewhere
    h = Holder(torch.from_numpy(s).to(dev))
    ok = True
    for t in range(x.shape[1] - 1):
        P1 = m.compute_pleasure(x[:, t])
        X = t_pleasure(m, o[:, t], sign_of(lk, dev))
        p1, Xh, frz = h(t, P1, X)
        for b in range(len(s)):
            if s[b] + SHOW <= t < s[b] + SHOW + FRZ:
                ok &= bool(torch.equal(p1[b], m.compute_pleasure(x[b:b + 1, s[b] + 2])[0]))
                ok &= bool(torch.equal(Xh[b], t_pleasure(m, o[b:b + 1, s[b] + 2], sign_of(lk[b:b + 1], dev))[0]))
            else:
                ok &= bool(torch.equal(p1[b], P1[b])) and bool(torch.equal(Xh[b], X[b]))
    out['c_B_holds_ok'] = ok
    out['c_swap_frames'] = s.tolist()
    out['c_pseudo_frames_in_20_80'] = bool(((freeze_frames(np.full(3000, -1)) >= 20) & (freeze_frames(np.full(3000, -1)) <= 80)).all())
    # (d) R25Model in the stage-I mode == the stage-I model (SuperpositionNetworkMotionPleasureHold, stage-I config)
    s1m = model_v12.SuperpositionNetworkMotionPleasureHold(s1model).to(dev)
    s1m.load_state_dict(torch.load(s1_path('p50', 0), map_location='cpu')['model'], strict=False)
    m0 = build_r25('p50', 0, dev)
    for mm in (s1m, m0):
        mm.eval()
    d1 = load_s1('test', 10)
    tl1 = loader.DataLoader(d1, 10, False, dev)
    cfg1 = s1_cfg()
    run1 = runners.PredictionFeaturePredictionRunner()
    with torch.no_grad():
        for seq, _ in tl1.load_sequence():
            seq = list(seq)
            torch.manual_seed(5)
            l_a = seq_stage1(m0, seq, 1, cfg1, run1)
            torch.manual_seed(5)
            l_b, _ = run1.run_seq(s1m, seq, 1, cfg1)
    out['d_stage1_mode_vs_stage1_model'] = {k: [float(l_a[k]), float(l_b[k])] for k in l_b}
    out['d_equal'] = all(abs(float(l_a[k]) - float(l_b[k])) <= 1e-6 * max(1.0, abs(float(l_b[k]))) for k in l_b)
    # (e) timings: one Sec.1 batch (100 episodes, 6 flows x inputs); one train2 iteration; one train32 iteration
    mp = build_plain('p50', 0, dev)
    dd, _, ss, lkk = load25('test', 100)
    t0 = time.time()
    sig_dataset(mp, dd, ss, lkk, dev)
    out['e_signal_sec_per_100_episodes'] = time.time() - t0
    m2 = build_r25('p50', 0, dev)
    for n, p in m2.named_parameters():
        p.requires_grad = n.startswith('p2_adapter')
    opt = R25.optimizer(m2, 1.0)
    out['e_train2_trainable'] = sorted(n for n, p in m2.named_parameters() if p.requires_grad)
    tl = loader.DataLoader({k: dd[k] for k in KEYS}, 10, True, dev)
    m2.train()
    ts = []
    for i, (seq, bidx) in enumerate(tl.load_sequence()):
        t0 = time.time()
        loss = seq_T(m2, seq, 1, exp_cfg, runner, 'B', torch.from_numpy(ss[bidx]).to(dev), sign_of(lkk[bidx], dev))
        opt.zero_grad()
        sum(loss.values()).backward()
        g = {n: float(p.grad.abs().sum()) for n, p in m2.named_parameters() if p.grad is not None}
        opt.step()
        ts.append(time.time() - t0)
        if i == 2:
            break
    out['e_train2_sec_per_iter'] = float(np.median(ts))
    out['e_train2_grad_params'] = sorted(g)
    t0 = time.time()
    out['e_eval2_keys'] = sorted(Eval2(dev, n=50).other(m2))
    out['e_eval2_sec_50'] = time.time() - t0
    os.makedirs(os.path.join(OUT, 'check'), exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'check', 'check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('check', 'signal', 'train2', 'data31', 'train32'))
    ap.add_argument('--jobs', nargs='*', default=[])
    ap.add_argument('--cond', choices=('none', 'p50', 'sr50'))
    ap.add_argument('--flow', choices=FLOWS)
    ap.add_argument('--pair', type=int, default=0)
    ap.add_argument('--ratio', type=float, default=0.1)
    a = ap.parse_args()
    if a.mode == 'check':
        check()
    elif a.mode == 'signal':
        signal(a.jobs)
    elif a.mode == 'train2':
        train2(a.cond, a.flow, a.pair)
    elif a.mode == 'data31':
        data31()
    else:
        train32(a.ratio)
