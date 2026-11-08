"""
Paper round 27 (paper/check/paper_instructions_round27.md; docs/paper_extra_log.md Sec.51; exploratory, outside the
pre-registration). Where the gain of the correct preference comes from (Sec.1, no training) and whether the loosest
upper bound moves the test quantity D_T (Sec.2, short training). GPU (GPU1-3). New code; existing models / data /
results (incl. round 26) are only read. Outputs: data/result/paper_extra/Round27/.

Terms T, W, Z, flows V / B, window: as round 26 (analyze/paper_extra/round26.py; log Sec.50.1-50.2).

Region of A-2 in A-1's view (Sec.1 item 3). A-2 is drawn black in A-1's view (black = every channel < 0.1 in [0, 1];
the calibration of round 21 / p2test_v11 uses the same rule). Checked on r25 test (100 episodes): A-2's bearing column
lies inside the black columns in 99.7% of the frames where black pixels exist; frames without black pixels (3.6%) all
have A-1 - A-2 distance <= 2.72 (A-2 too close to be drawn). For
the step that predicts frame t+1 from frame t, the region is the set of view columns (all 16 rows) in which A-2's black
pixels appear in the TRUE frame t or t+1, widened by DIL = 2 columns on each side (circular; 1 column = 0.098 rad).
L1 of A-1's view is split additively: L1 = L1_A2 + L1_rest, both = sum of |pred - truth| over the pixels of the part /
(number of all pixels), so the two parts add up to the runner's L1 exactly.

Modes
  check   (a) Sec.1 recomputation reproduces round 26's saved window sums (p50 pair 1, first 100 r25 episodes, all flows
          and inputs; L1_A2 + L1_rest = L1); region sanity (A-2's bearing column inside the black columns; region share);
          (b) stage-I mode of the round-27 model == each stage-I model on its stage-I data (none: v11_s1_random with
          v12_s1_random_l1_s0; p50: v11_s1_mixed with v12_s1_mixed_l1_s0; sr50: v11_s1_mixed + signs with the signed
          model paper_extra/E1_signrand50/s1_signrand50_l1_s0); (c) Sec.2 epoch-0 D_T == round 26 s2_<cond>_B_s0 ep050
          (first 50 test episodes); (d) gradient only to the adapter and the shared LSTM; timings.
  sig     Sec.1: stage-I predictors (none / p50 / sr50 / p80, pairs 1-6), no adapter; per episode window sums of
          [L1_A2, L1_rest, feature_prediction_self, feature_prediction_other] for X in {T, W, Z} x flow {V, B}; data
          paper_r25_swap50 train (3000) and round 20 S / O (s = 50, 600 each); also the region share.
          -> Round27/signal/<cond>_s<pair>.npz
  train   Sec.2: start = round 26 Sec.2 B end (Round26/runs/s2_<cond>_B_s<pair>/model/050.pth: the adapter) on the
          stage-I predictor of the pair; G replaced by T (p2 = [0, 0, mask(A(T)), 1]); trainable: the adapter (lr 1e-3)
          and the shared LSTM (superposition_module, lr 1e-4 = 1/10); AdamW wd 0.01 (bias 0). Every batch = 10 episodes
          of paper_r25_swap50 train in flow B (stage-II loss) + 10 episodes of the condition's own stage-I training data
          in the stage-I flow and loss (l1); loss = 0.5 (L_II + L_I). 50 epochs. Evaluation every 5 epochs incl. 0:
          D_T (round 26 Eval2), C_self (round 25), L1 of A-1's view on paper_r25_swap50 test (300) with T through the
          adapter in flow V and in flow B; at epochs 0 and 50 the Sec.1 window sums (flow B, T / W through the adapter
          and mask, Z = zeros fed unchanged) on r25 train (3000, main) and r25 test (300, reference).
          -> Round27/runs/s2_<cond>_s<pair>/
Usage: python analyze/paper_extra/round27.py check | sig --jobs p50:0 ... | train --cond p50 --pair 0
"""

import argparse
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
import round26 as R26  # noqa
import round25_train as R25  # noqa
import round21_eval as R21  # noqa  (registers the signed classes)
import round24_eval as R24E  # noqa
import util  # noqa
from model import model_v12  # noqa
from exp import runner as runners, loader  # noqa

RES, _PI = R25.RES, R25._PI
OUT = os.path.join(RES, 'paper_extra', 'Round27')
R26OUT = R26.OUT
DIL = 2
WIN = R26.WIN
INPUTS, FLOWS = R26.INPUTS, R26.FLOWS
DS1 = {'p50': os.path.join(_PI, 'data', 'data', 'v11_s1_mixed', 'data.h5'),
       'sr50': os.path.join(_PI, 'data', 'data', 'v11_s1_mixed', 'data.h5'),
       'none': os.path.join(_PI, 'data', 'data', 'v11_s1_random', 'data.h5')}
SIGNS = os.path.join(_PI, 'data', 'data', 'paper_s1_mixed50_signrand', 'signs.h5')
S1CFG = {'p50': 'v12_s1_mixed_l1_s0.yml', 'none': 'v12_s1_random_l1_s0.yml',
         'sr50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s0.yml'}
COMP = ('l1_a2', 'l1_rest', 'feat_self', 'feat_other')


# ---------------------------------------------------------------- model
class R27Model(R25.R25Model):
    """R25Model (variant M) + the stage-I sign of the sign-random condition: while `_sign` is set (stage-I half of an
    sr50 batch only), process-1's pleasure is multiplied by it, exactly as model_v14.SuperpositionNetworkMotionPleasure
    HoldSigned does with a1_liked_t (sign = 2 a1_liked - 1). `_sign` is None everywhere else."""

    def __init__(self, config, mask=True):
        super().__init__(config, mask)
        self._sign = None

    def compute_pleasure(self, sv):
        P = super().compute_pleasure(sv)
        return P if self._sign is None else P * self._sign


def build27(cond, pair, dev, adapter_from=None):
    _, mcfg, _ = R25.configs()
    util.seed_all(0)
    m = R27Model(mcfg, True)
    res = m.load_state_dict(torch.load(R26.s1_path(cond, pair), map_location='cpu')['model'], strict=False)
    assert not res.unexpected_keys
    assert all(k.startswith(('motion_generator_module', 'p2_adapter', 'p2_mask')) for k in res.missing_keys)
    if adapter_from:
        st = torch.load(adapter_from, map_location='cpu')
        assert sorted(st) == ['p2_adapter.bias', 'p2_adapter.weight'], sorted(st)
        m.load_state_dict(st, strict=False)
    return m.to(dev)


def r26_adapter(cond, pair):
    return os.path.join(R26OUT, 'runs', f's2_{cond}_B_s{pair}', 'model', '050.pth')


# ---------------------------------------------------------------- Sec.1
def region_cols(xa, xb):
    """xa, xb: scaled true frames (B, 3, 16, 64). -> (B, 64) bool: columns with black pixels in either, widened."""
    blk = (((xa + 1) / 2).amax(1) < 0.1).any(1) | (((xb + 1) / 2).amax(1) < 0.1).any(1)   # (B, 64)
    out = blk.clone()
    for d in range(1, DIL + 1):
        out |= torch.roll(blk, d, 1) | torch.roll(blk, -d, 1)
    return out


@torch.no_grad()
def window_losses(m, x, o, mo, sign, s, flow, inp):
    """As round26.window_losses, but L1 split into the A-2 region and the rest. -> (B, 4) window sums of
    [L1_A2, L1_rest, feature_prediction_self, feature_prediction_other] and (B,) window sum of the region share.
    For an R25Model, T / W go through the adapter and mask; Z is fed unchanged (p2_bypass), as round 26 Eval2."""
    B = x.shape[0]
    tmax = int(s.max()) + WIN
    m.init_state(B)
    hold = R26.Holder(s)
    out = torch.zeros(B, 4, device=x.device)
    share = torch.zeros(B, device=x.device)
    has_bypass = hasattr(m, 'p2_bypass')
    if has_bypass:
        m.p2_bypass = (inp == 'Z')
    try:
        for t in range(tmax):
            xt = {'self_vision': x[:, t], 'self_motion': mo[:, t]}
            X = R26.t_pleasure(m, o[:, t], sign) * R26.SIGN[inp]
            p1 = None
            if flow == 'B':
                p1, X, _ = hold(t, m.compute_pleasure(x[:, t]), X)
            p = 0 if t == 0 else 1
            pred = m(xt, p, p, p1_override=p1, p2_override=R26.p2_of(X))
            inw = (t >= s) & (t < s + WIN)
            if inw.any():
                tr = x[:, t + 1]
                err = (pred['self_vision'] - tr).abs()                                  # (B, 3, 16, 64)
                reg = region_cols(x[:, t], tr).float()                                  # (B, 64)
                npix = float(err[0].numel())
                l1a = (err * reg[:, None, None, :]).flatten(1).sum(1) / npix
                l1r = (err * (1 - reg)[:, None, None, :]).flatten(1).sum(1) / npix
                st = m.get_state()
                fs = ((m.predict_feature(st['self'][0]) - m.self_vision_encoder_module(tr)) ** 2).flatten(1).mean(1)
                fo = ((m.predict_feature(st['other'][0]) - m.other_vision_encoder_module(tr)) ** 2).flatten(1).mean(1)
                w = inw.float()
                out += torch.stack([l1a, l1r, fs, fo], 1) * w[:, None]
                share += reg.mean(1) * w
    finally:
        if has_bypass:
            m.p2_bypass = False
    return out.cpu().numpy(), share.cpu().numpy()


def sig_dataset(m, D, s, lk, dev, flows=FLOWS, batch=100):
    res = {f'{f}_{i}': [] for f in flows for i in INPUTS}
    shares = []
    N = D['self_vision'].shape[0]
    for b0 in range(0, N, batch):
        sl = slice(b0, b0 + batch)
        sb = torch.from_numpy(s[sl]).to(dev)
        tn = int(s[sl].max()) + WIN + 1
        x = R21.scaled(D['self_vision'][sl, :tn], dev)
        o = R21.scaled(D['other_vision'][sl, :tn], dev)
        mo = torch.from_numpy(D['self_motion'][sl, :tn]).float().to(dev)
        sg = R26.sign_of(lk[sl], dev)
        for f in flows:
            for i in INPUTS:
                v, sh = window_losses(m, x, o, mo, sg, sb, f, i)
                res[f'{f}_{i}'].append(v)
                if f == flows[0] and i == 'T':
                    shares.append(sh)
    out = {k: np.concatenate(v).astype(np.float32) for k, v in res.items()}
    out['region_share'] = np.concatenate(shares).astype(np.float32)
    return out


def sig(jobs):
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    od = os.path.join(OUT, 'signal')
    os.makedirs(od, exist_ok=True)
    d25, sw, s25, lk25 = R26.load25('train')
    r20 = {v: R24E.load(v) for v in ('S', 'O')}
    for job in jobs:
        cond, pair = job.split(':')
        pair = int(pair)
        path = os.path.join(od, f'{cond}_s{pair}.npz')
        assert not os.path.exists(path), path
        t0 = time.time()
        m = R26.build_plain(cond, pair, dev)
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
def s1_cfg(cond):
    return util.load_config(os.path.join(_PI, 'config', 'exp', S1CFG[cond]))


def load_s1(cond, split='train', n=0):
    g = h5py.File(DS1[cond], 'r')[split]
    sl = slice(0, n or None)
    d = {k: g[k][sl] for k in ('self_vision', 'self_motion', 'self_position', 'other_motion', 'other_position')}
    liked = None
    if cond == 'sr50':
        with h5py.File(SIGNS, 'r') as f:
            assert f.attrs['source'] == 'v11_s1_mixed'
            liked = f[split]['a1_liked'][()][sl].astype(np.int64)
        assert len(liked) == d['self_position'].shape[0]
    return d, liked


def seq_stage1(m, seq, ep, cfg, runner, liked_b=None):
    """round26.seq_stage1 (stage-I flow and l1 loss) with, for sr50, process-1's pleasure multiplied by the
    episode's sign 2 a1_liked - 1 (as the signed stage-I model)."""
    if liked_b is not None:
        m._sign = (2 * torch.as_tensor(liked_b, device=next(m.parameters()).device).float() - 1)[:, None]
    try:
        return R26.seq_stage1(m, seq, ep, cfg, runner)
    finally:
        m._sign = None


class Eval27:
    def __init__(self, dev, n=0):
        self.dev = dev
        self.e2 = R26.Eval2(dev, n)
        self.e25 = R25.Evaluator(dev, n)
        self.dte, _, self.ste, self.lkte = R26.load25('test', n)
        self.dtr, _, self.str_, self.lktr = R26.load25('train', n * 10)
        self.exp_cfg, _, _ = R25.configs()

    @torch.no_grad()
    def l1_T(self, m, flow, batch=50):
        """Mean L1 of A-1's predicted view over all steps and episodes of r25 test, T through the adapter (training
        flow: vision masked for t > 0; flow B holds as in training)."""
        m.eval()
        runner = runners.MotionGenerationPredictionFeaturePredictionRunner()
        D, s, lk = self.dte, self.ste, self.lkte
        tot, n = 0.0, 0
        for b0 in range(0, D['self_vision'].shape[0], batch):
            x = R21.scaled(D['self_vision'][b0:b0 + batch], self.dev)
            o = R21.scaled(D['other_vision'][b0:b0 + batch], self.dev)
            mo = torch.from_numpy(D['self_motion'][b0:b0 + batch]).float().to(self.dev)
            sg = R26.sign_of(lk[b0:b0 + batch], self.dev)
            B = x.shape[0]
            m.init_state(B)
            hold = R26.Holder(torch.from_numpy(s[b0:b0 + batch]).to(self.dev))
            for t in range(x.shape[1] - 1):
                X = R26.t_pleasure(m, o[:, t], sg)
                p1 = None
                if flow == 'B':
                    p1, X, _ = hold(t, m.compute_pleasure(x[:, t]), X)
                p = runner.gen_mask_prob(t, self.exp_cfg)
                pred = m({'self_vision': x[:, t], 'self_motion': mo[:, t]}, p, p, p1_override=p1,
                         p2_override=R26.p2_of(X))
                tot += float(F.l1_loss(pred['self_vision'], x[:, t + 1])) * B
                n += B
        return tot / n

    def sig_B(self, m):
        m.eval()
        out = {}
        for tag, D, s, lk in (('tr', self.dtr, self.str_, self.lktr), ('te', self.dte, self.ste, self.lkte)):
            for k, v in sig_dataset(m, D, s, lk, self.dev, flows=('B',)).items():
                out[f'{tag}_{k}'] = v
        return out


def train(cond, pair, n_ep=50, every=5, n_eval=0):
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    name = f's2_{cond}_s{pair}'
    od = os.path.join(OUT, 'runs', name)
    assert not os.path.exists(od), od
    for sub in ('model', 'eval'):
        os.makedirs(os.path.join(od, sub))
    exp_cfg, _, _ = R25.configs()
    m = build27(cond, pair, dev, r26_adapter(cond, pair))
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith(('p2_adapter', 'superposition_module'))
    opt = R25.optimizer(m, 0.1)
    d, sw, s, lk = R26.load25('train')
    tl2 = loader.DataLoader(d, exp_cfg.train.batch_size, True, dev)
    cfg1 = s1_cfg(cond)
    d1, liked1 = load_s1(cond, 'train')
    tl1 = loader.DataLoader(d1, cfg1.train.batch_size, True, dev)
    ev = Eval27(dev, n_eval)
    run2 = runners.MotionGenerationPredictionFeaturePredictionRunner()
    run1 = runners.PredictionFeaturePredictionRunner()
    log = {'name': name, 'cond': cond, 'pair': pair, 'start_predictor': R26.s1_path(cond, pair),
           'start_adapter': r26_adapter(cond, pair), 'stage1_data': DS1[cond], 'stage1_signs': SIGNS if liked1 is not None else None,
           'stage1_cfg': S1CFG[cond], 'trainable': sorted(n for n, p in m.named_parameters() if p.requires_grad),
           'epochs': []}

    def evaluate(ep):
        t0 = time.time()
        sav = ev.e2.other(m)
        sav['c_self'] = ev.e25.c_self(m)
        sav['l1_T_V'] = np.float32(ev.l1_T(m, 'V'))
        sav['l1_T_B'] = np.float32(ev.l1_T(m, 'B'))
        np.savez_compressed(os.path.join(od, 'eval', f'ep{ep:03d}.npz'), **sav)
        if ep in (0, n_ep):
            np.savez_compressed(os.path.join(od, 'eval', f'sig_ep{ep:03d}.npz'), **ev.sig_B(m))
        torch.save({k: v for k, v in m.state_dict().items() if k.startswith(('p2_adapter', 'superposition_module'))},
                   os.path.join(od, 'model', f'{ep:03d}.pth'))
        m.train()
        return time.time() - t0

    log['eval0_sec'] = evaluate(0)
    json.dump(log, open(os.path.join(od, 'log.json'), 'w'), indent=1)
    print(json.dumps({'epoch': 0, 'eval_sec': round(log['eval0_sec'], 1)}), flush=True)
    util.seed_all(R25.SEED)
    m.train()
    for ep in range(1, n_ep + 1):
        t0 = time.time()
        acc, nb = defaultdict(float), 0
        for (seq2, b2), (seq1, b1) in zip(tl2.load_sequence(), tl1.load_sequence()):
            l2 = R26.seq_T(m, seq2, ep, exp_cfg, run2, 'B', torch.from_numpy(s[b2]).to(dev), R26.sign_of(lk[b2], dev))
            l1 = seq_stage1(m, seq1, ep, cfg1, run1, None if liked1 is None else liked1[b1])
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
            rec['eval_sec'] = evaluate(ep)
        log['epochs'].append(rec)
        json.dump(log, open(os.path.join(od, 'log.json'), 'w'), indent=1)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    json.dump({'finished': time.strftime('%Y-%m-%d %H:%M')}, open(os.path.join(od, 'done.json'), 'w'))


# ---------------------------------------------------------------- check
def check():
    from model import model_v14  # noqa
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    out = {}
    # (a) Sec.1 recomputation vs round 26 (p50 pair 1, first 100 r25 train episodes)
    t0 = time.time()
    d25, sw, s25, lk25 = R26.load25('train', 100)
    m = R26.build_plain('p50', 0, dev)
    new = sig_dataset(m, d25, s25, lk25, dev)
    out['a_sec_per_100_episodes'] = time.time() - t0
    z = np.load(os.path.join(R26OUT, 'signal', 'p50_s0.npz'))
    diffs = {}
    for f in FLOWS:
        for i in INPUTS:
            o = z[f'r25_{f}_{i}'][:100].astype(np.float64)
            n_ = new[f'{f}_{i}'].astype(np.float64)
            mine = np.stack([n_[:, 0] + n_[:, 1], n_[:, 2], n_[:, 3]], 1)
            diffs[f'{f}_{i}'] = float(np.max(np.abs(mine - o) / np.maximum(np.abs(o), 1e-6)))
    out['a_rel_diff_vs_round26'] = diffs
    out['a_max_rel_diff'] = max(diffs.values())
    out['a_region_share_mean_per_step'] = float(new['region_share'].mean() / WIN)
    out['a_l1_a2_share_of_l1_T_V'] = float(new['V_T'][:, 0].sum() / (new['V_T'][:, 0] + new['V_T'][:, 1]).sum())
    # region sanity on r25 test frames: bearing column of A-2 inside the black columns when A-2 is drawn
    from v11core.pleasure import COL_ANGLES
    g = h5py.File(R26.D25, 'r')['test']
    sv = g['self_vision'][:100].astype(np.float32) / 255.0
    sp, op = g['self_position'][:100].astype(np.float64), g['other_position'][:100].astype(np.float64)
    blkc = (sv.max(-1) < 0.1).any(-2)                                        # (N, T, 64)
    dvec = op - sp
    bear = np.arctan2(dvec[..., 1], dvec[..., 0])
    col = np.argmin(np.abs(np.angle(np.exp(1j * (COL_ANGLES[None, None] - bear[..., None])))), -1)
    vis = blkc.any(-1)
    dist = np.linalg.norm(dvec, axis=-1)
    out['a_frames_with_A2_drawn'] = float(vis.mean())
    out['a_bearing_col_in_black_cols'] = float(np.take_along_axis(blkc, col[..., None], -1)[..., 0][vis].mean())
    out['a_max_dist_when_not_drawn'] = float(dist[~vis].max()) if (~vis).any() else None
    out['a_min_dist_when_drawn'] = float(dist[vis].min())
    # (b) stage-I mode == stage-I model, per condition
    s1mcfg = util.load_config(os.path.join(_PI, 'config', 'model', 'SuperpositionNetworkMotionPleasureHold', 'default.yml'))
    run1 = runners.PredictionFeaturePredictionRunner()
    for cond in ('none', 'p50', 'sr50'):
        cls = model_v14.SuperpositionNetworkMotionPleasureHoldSigned if cond == 'sr50' else model_v12.SuperpositionNetworkMotionPleasureHold
        ref = cls(s1mcfg).to(dev)
        ref.load_state_dict(torch.load(R26.s1_path(cond, 0), map_location='cpu')['model'], strict=False)
        m0 = build27(cond, 0, dev)
        for mm in (ref, m0):
            mm.eval()
        d1, liked = load_s1(cond, 'test', 10)
        cfg1 = s1_cfg(cond)
        with torch.no_grad():
            seq = list(loader.DataLoader(d1, 10, False, dev).load_sequence())[0][0]
            seq = list(seq)
            torch.manual_seed(5)
            la = seq_stage1(m0, seq, 1, cfg1, run1, liked)
            torch.manual_seed(5)
            if cond == 'sr50':
                lt = torch.from_numpy(np.repeat(liked[:, None], d1['self_position'].shape[1], 1)).to(dev)
                seq_l = [({**x, 'a1_liked_t': lt[:, t]}, y, t) for x, y, t in seq]
                lb, _ = run1.run_seq(ref, seq_l, 1, cfg1)
            else:
                lb, _ = run1.run_seq(ref, seq, 1, cfg1)
            # sr50 without the sign must differ (the sign matters)
            if cond == 'sr50':
                torch.manual_seed(5)
                lc = seq_stage1(m0, seq, 1, cfg1, run1, None)
                out['b_sr50_unsigned_differs'] = any(abs(float(lc[k]) - float(lb[k])) > 1e-4 for k in lb)
        out[f'b_{cond}_stage1_mode_vs_model'] = {k: [float(la[k]), float(lb[k])] for k in lb}
        out[f'b_{cond}_equal'] = all(abs(float(la[k]) - float(lb[k])) <= 1e-5 * max(1.0, abs(float(lb[k]))) for k in lb)
    # (c) Sec.2 epoch-0 D_T == round 26 s2_p50_B_s0 ep050 (first 50 test episodes)
    m2 = build27('p50', 0, dev, r26_adapter('p50', 0))
    e2 = R26.Eval2(dev, n=50).other(m2)
    zr = np.load(os.path.join(R26OUT, 'runs', 's2_p50_B_s0', 'eval', 'ep050.npz'))
    out['c_ep0_vs_round26_ep050_max_abs_diff'] = {k: float(np.nanmax(np.abs(e2[k] - zr[k][:50])))
                                                   for k in ('S_T_delta', 'O_T_delta', 'S_Z_delta', 'O_Z_delta')}
    out['c_adapter_equal'] = bool(np.array_equal(e2['adapter_w'], zr['adapter_w']))
    # (d) gradients and timings
    for n, p in m2.named_parameters():
        p.requires_grad = n.startswith(('p2_adapter', 'superposition_module'))
    opt = R25.optimizer(m2, 0.1)
    out['d_param_groups_lr'] = sorted({(g_['lr'], g_['weight_decay'], len(g_['params'])) for g_ in opt.param_groups})
    exp_cfg, _, _ = R25.configs()
    dd, _, ss, lkk = R26.load25('test', 100)
    tl2 = loader.DataLoader({k: dd[k] for k in R26.KEYS}, 10, True, dev)
    d1, liked1 = load_s1('sr50', 'test', 100)
    tl1 = loader.DataLoader(d1, 10, True, dev)
    cfg1 = s1_cfg('sr50')
    run2 = runners.MotionGenerationPredictionFeaturePredictionRunner()
    m2.train()
    ts = []
    for i, ((seq2, b2), (seq1, b1)) in enumerate(zip(tl2.load_sequence(), tl1.load_sequence())):
        t0 = time.time()
        l2 = R26.seq_T(m2, seq2, 1, exp_cfg, run2, 'B', torch.from_numpy(ss[b2]).to(dev), R26.sign_of(lkk[b2], dev))
        l1 = seq_stage1(m2, seq1, 1, cfg1, run1, liked1[b1])
        opt.zero_grad()
        (0.5 * (sum(l2.values()) + sum(l1.values()))).backward()
        g = sorted({n.split('.')[0] for n, p in m2.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0})
        opt.step()
        ts.append(time.time() - t0)
        if i == 2:
            break
    out['d_modules_with_grad'] = g
    out['d_sec_per_iter'] = float(np.median(ts))
    t0 = time.time()
    ev = Eval27(dev, n=20)
    out['d_l1_T_V_20'] = ev.l1_T(m2, 'V')
    out['d_l1_T_B_20'] = ev.l1_T(m2, 'B')
    out['d_c_self_20_mean'] = float(np.nanmean(ev.e25.c_self(m2)))
    out['d_eval_sec_20'] = time.time() - t0
    os.makedirs(os.path.join(OUT, 'check'), exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'check', 'check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('check', 'sig', 'train'))
    ap.add_argument('--jobs', nargs='*', default=[])
    ap.add_argument('--cond', choices=('none', 'p50', 'sr50'))
    ap.add_argument('--pair', type=int, default=0)
    a = ap.parse_args()
    if a.mode == 'check':
        check()
    elif a.mode == 'sig':
        sig(a.jobs)
    else:
        train(a.cond, a.pair)
