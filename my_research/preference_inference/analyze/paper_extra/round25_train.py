"""
Paper round 25 Sec.1 (paper/check/paper_instructions_round25.md; docs/paper_extra_log.md Sec.49; trial on pair 1 before
the advisor's approval; exploratory, outside the pre-registration). GPU (GPU1-3). New code; existing models / data /
results are only read. Outputs: data/result/paper_extra/Round25/runs/<run>/.

Model R25Model = stage-II model (model_v12.SuperpositionNetworkMotionPleasureHoldOther: stage-I predictor + G) with
  * adapter A: nn.Linear(8, 8) on the other pathway's pleasure input, weight = identity, bias = 0 at the start;
  * mask (variant M): after A, just before the LSTM, ranges 2-5 set to 0 (variant N: no mask);
  * p2 = [0, 0, mask(A(P2)), 1], P2 = G's output (or an override's pleasure part); `p2_bypass` (evaluation only)
    feeds an override unchanged (no A, no mask).
Start: stage-I predictor of pair 1 (seed 0, l1 stage ep200): none v12_s1_random_l1_s0, p50 v12_s1_mixed_l1_s0,
sr50 paper_extra/E1_signrand50/s1_signrand50_l1_s0 (loaded strict=False: G and A are the only missing keys). G is
newly initialised: util.seed_all(0) before the model is built, so G's initial weights are identical in the three
conditions (checked). Loss, runner, batch 10, AdamW lr 1e-3 / weight decay 0.01 (0 for biases), vision masked for
t > 0 = stage II (config v12_s2_pref_mixed_s0.yml, MotionGenerationPredictionFeaturePredictionRunner). Training data:
data/data/paper_r25_swap50/data.h5 'train' (collect_data_round24.py: p_swap 0.5, swap frame 20-80, p_rule 0.5, liked
colour 1/2; 3000 episodes; A-2's camera view is not fed). util.seed_all(0) again just before epoch 1.
  stage 1  trainable A + G, 100 epochs; variants M and N.
  stage 2  from stage-1 M ep100; trainable A + G + the shared LSTM (superposition_module), LSTM lr = ratio x 1e-3,
           ratio 0.1 or 0.01; 50 epochs; fresh optimizer.
Checkpoints (model state) and evaluations every 5 epochs including epoch 0:
  other  round 24 Sec.1.5 test: round 20 data p_rule 0.8, T_s 50, variants S and O, frames 0..52 shown, frames 53..62
         frozen (vision masked, motion hidden, p1 = 0); P2 = G: G's output at step 52 held (through A and the mask, as
         in training); Z: p2 = [0, 0, 0 x 8, 1] fed unchanged (p2_bypass). Delta = s_Lold - s_Lnew from the predicted
         image (read_a2), L_old / L_new = A-2's liked colour in the pre- / post-swap layout; per episode. Also mean
         |G output| and |mask(A(G output))| per range at step 52 (variant S), by liked colour.
  self   (stage 1: epochs 0 and 100; stage 2: every 5 epochs) (a) L1 of the predicted A-1 view on the 300 'test'
         episodes of the new data in the training flow; (b) round 21's C_self (hold): V1 freeze of round21_eval with
         p2_override = zeros(11) fed unchanged (= the stage-I model's process-2) and the stage-I motion-masking
         schedule switched on; colour-agnostic d*; C_self = Delta(noswap) - Delta(swap) per episode.
Usage: python analyze/paper_extra/round25_train.py stage1 --cond p50 --variant M
       python analyze/paper_extra/round25_train.py stage2 --cond p50 --ratio 0.1
       python analyze/paper_extra/round25_train.py check
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
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round21_eval as R21  # noqa  (registers the signed classes; freeze tools, data, views)
import round24_eval as R24E  # noqa  (round-20 data, targets)
from p2test_v11 import read_a2  # noqa
import util  # noqa
from model import model_v12  # noqa
from model.model_v12 import MOTION_DIM  # noqa
from model import util as mutil  # noqa
from exp import runner as runners, loader  # noqa

_PI = R21._PI
RES = R21.RES
OUT = os.path.join(RES, 'paper_extra', 'Round25')
DATA = os.path.join(_PI, 'data', 'data', 'paper_r25_swap50', 'data.h5')
START = {'none': 'v12_s1_random_l1_s0', 'p50': 'v12_s1_mixed_l1_s0',
         'sr50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s0'}
SEED = 0
EP1, EP2, EVERY = 100, 50, 5
LR, WD = 1e-3, 0.01
COLOUR_BINS = (0, 1, 6, 7)


class R25Model(model_v12.SuperpositionNetworkMotionPleasureHoldOther):
    def __init__(self, config, mask=True):
        super().__init__(config)
        self.p2_adapter = nn.Linear(8, 8)
        with torch.no_grad():
            self.p2_adapter.weight.copy_(torch.eye(8))
            self.p2_adapter.bias.zero_()
        m = torch.zeros(8)
        m[list(COLOUR_BINS)] = 1.0
        self.register_buffer('p2_mask', m if mask else torch.ones(8))
        self.p2_bypass = False

    def p2_pleasure(self, P):
        return self.p2_adapter(P) * self.p2_mask

    def forward(self, x, p_mask_vision_self, p_mask_vision_other, p1_override=None, p2_override=None,
                hidden_override=None):
        sv = x['self_vision']
        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)
        p1 = self.process1_input(x, p1_override, hidden_override)
        q2 = self.motion_generator_module(ov_enc)
        if p2_override is None:
            p2 = torch.cat([torch.zeros_like(q2[:, :MOTION_DIM]), self.p2_pleasure(q2), torch.ones_like(q2[:, :1])], 1)
        elif self.p2_bypass:
            p2 = p2_override
        else:
            p2 = torch.cat([p2_override[:, :2], self.p2_pleasure(p2_override[:, 2:10]), p2_override[:, 10:]], 1)
        self._t += 1
        sv_enc = mutil.mask(sv_enc, p_mask_vision_self)
        ov_enc = mutil.mask(ov_enc, p_mask_vision_other)
        ss, os_ = self.superposition_module(sv_enc, p1, ov_enc, p2)
        so = self.integration_module(F.dropout(ss, p=0.5, training=self.training),
                                     F.dropout(os_, p=0.5, training=self.training))
        return {'self_vision': self.vision_decoder_module(so), 'q2_hat': q2}


def configs():
    exp_cfg = util.load_config(os.path.join(_PI, 'config', 'exp', 'v12_s2_pref_mixed_s0.yml'))
    mcfg = util.load_config(os.path.join(_PI, 'config', 'model', exp_cfg.model.name, exp_cfg.model.config + '.yml'))
    s1 = util.load_config(os.path.join(_PI, 'config', 'model', 'SuperpositionNetworkMotionPleasureHold', 'default.yml'))
    return exp_cfg, mcfg, s1


def build(cond, mask, dev):
    exp_cfg, mcfg, _ = configs()
    util.seed_all(SEED)
    m = R25Model(mcfg, mask)
    st = torch.load(os.path.join(RES, START[cond], str(SEED), 'model', '00200.pth'), map_location='cpu')['model']
    res = m.load_state_dict(st, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    assert all(k.startswith(('motion_generator_module', 'p2_adapter', 'p2_mask')) for k in res.missing_keys), res.missing_keys
    return m.to(dev), exp_cfg


def load_split(split, n=0):
    g = h5py.File(DATA, 'r')[split]
    sl = slice(0, n or None)
    return {k: g[k][sl] for k in ('self_vision', 'self_motion', 'self_position', 'other_motion', 'other_position')}


def optimizer(m, ratio):
    groups = defaultdict(list)
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        lr = LR * (ratio if n.startswith('superposition_module') else 1.0)
        wd = 0.0 if 'bias' in n else WD
        groups[(lr, wd)].append(p)
    return torch.optim.AdamW([{'params': v, 'lr': lr, 'weight_decay': wd} for (lr, wd), v in groups.items()])


def set_trainable(m, stage):
    keep = ('p2_adapter', 'motion_generator_module') + (('superposition_module',) if stage == 2 else ())
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith(keep)


# ---------------------------------------------------------------- evaluations
class Evaluator:
    def __init__(self, dev, n=0):
        self.dev = dev
        self.ab = R21.calib_common()
        self.r20 = {v: R24E.load(v) for v in ('S', 'O')}
        if n:
            self.r20 = {v: {k: a[:n] for k, a in D.items()} for v, D in self.r20.items()}
        self.lo, self.ln = R24E.targets(self.r20['S']['layout'], self.r20['S']['a2_liked'])
        self.test = loader.DataLoader(load_split('test', n), 10, False, dev)
        self.r21 = {v: R21.load_data(v, n) for v in ('swap', 'noswap')}
        self.views = R21.load_views(n)
        _, _, s1 = configs()
        self.mm = s1.motion_mask

    @torch.no_grad()
    def other(self, m, batch=50):
        m.eval()
        T0, K = R24E.T0, R24E.K
        out = {}
        for v, D in self.r20.items():
            N = D['self_vision'].shape[0]
            a2 = D['other_position'][:, T0 - 1].astype(np.float64)
            sp = D['self_position'][:, T0 - 1:T0 - 1 + K].astype(np.float64)
            preds = {'G': [], 'Z': []}
            qraw, qin = [], []
            for b0 in range(0, N, batch):
                x = R21.scaled(D['self_vision'][b0:b0 + batch, :T0 + K], self.dev)
                mo = torch.from_numpy(D['self_motion'][b0:b0 + batch]).float().to(self.dev)
                B = x.shape[0]
                for inp in ('G', 'Z'):
                    util.seed_all(0)
                    m.init_state(B)
                    m.p2_bypass = False
                    for t in range(T0):
                        o = m({'self_vision': x[:, t], 'self_motion': mo[:, t]}, 0, 0)
                    q2 = o['q2_hat']
                    Q = torch.zeros(B, 11, device=self.dev)
                    Q[:, 10] = 1.0
                    if inp == 'G':
                        Q[:, 2:10] = q2
                        if v == 'S':
                            qraw.append(q2.abs().cpu().numpy())
                            qin.append(m.p2_pleasure(q2).abs().cpu().numpy())
                    else:
                        m.p2_bypass = True
                    P1 = torch.zeros(B, 8, device=self.dev)
                    hid = torch.ones(B, dtype=torch.bool, device=self.dev)
                    pr = []
                    for k in range(K):
                        o = m({'self_vision': x[:, T0 + k], 'self_motion': mo[:, T0 + k]}, 1.0, 1.0, p1_override=P1,
                              p2_override=Q, hidden_override=hid)
                        pr.append(((o['self_vision'] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
                    m.p2_bypass = False
                    preds[inp].append(np.stack(pr, 1))
            for inp, pl in preds.items():
                pr = np.concatenate(pl)
                est = np.stack([read_a2(pr[:, i], sp[:, i], *self.ab) for i in range(K)], 1)
                S = R21.other_slopes(est, a2, self.lo, self.ln)
                out[f'{v}_{inp}_delta'] = (S[:, 0] - S[:, 1]).astype(np.float32)
                out[f'{v}_{inp}_readable'] = np.float32(np.isfinite(est[..., 0]).mean())
            if v == 'S':
                out['q2_abs_raw'] = np.concatenate(qraw).astype(np.float32)
                out['q2_abs_in'] = np.concatenate(qin).astype(np.float32)
        out['a2_liked'] = self.r20['S']['a2_liked']
        return out

    @torch.no_grad()
    def self_l1(self, m):
        m.eval()
        runner = runners.MotionGenerationPredictionFeaturePredictionRunner()
        exp_cfg, _, _ = configs()
        tot, n = 0.0, 0
        for seq, _ in self.test.load_sequence():
            for x, truth, t in seq:
                if t == 0:
                    m.init_state(x['self_vision'].shape[0])
                p = runner.gen_mask_prob(t, exp_cfg)
                pred = m(x, p, p)
                tot += float(F.l1_loss(pred['self_vision'], truth['self_vision'])) * x['self_vision'].shape[0]
                n += x['self_vision'].shape[0]
        return tot / n

    @torch.no_grad()
    def c_self(self, m, batch=50):
        m.eval()
        save = (m.motion_masking, m.len_min, m.len_max, m.n_steps)
        m.motion_masking, m.len_min, m.len_max, m.n_steps = True, int(self.mm.len_min), int(self.mm.len_max), int(self.mm.n_steps)
        m.p2_bypass = True
        T0, K = 23, 15
        res = {}
        try:
            for d, D in self.r21.items():
                N = D['self_vision'].shape[0]
                outp = []
                for b0 in range(0, N, batch):
                    x = R21.scaled(D['self_vision'][b0:b0 + batch], self.dev)
                    mo = torch.from_numpy(D['self_motion'][b0:b0 + batch]).float().to(self.dev)
                    B = x.shape[0]
                    util.seed_all(0)
                    m.init_state(B)
                    Z = torch.zeros(B, 11, device=self.dev)
                    for t in range(T0):
                        m({'self_vision': x[:, t], 'self_motion': mo[:, t]}, 0, 0, p2_override=Z)
                    P1 = m.compute_pleasure(x[:, T0 - 1])
                    hid = torch.ones(B, dtype=torch.bool, device=self.dev)
                    pr = []
                    for k in range(K):
                        o = m({'self_vision': x[:, T0 + k], 'self_motion': mo[:, T0 + k]}, 1.0, 1.0, p1_override=P1,
                              p2_override=Z, hidden_override=hid)
                        pr.append(((o['self_vision'] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
                    outp.append(np.stack(pr, 1))
                ds = R21.dstar_lt(np.concatenate(outp), self.views['views_lt'][:, 1], self.views['valid_lt'][:, 1], None)
                sl = R21.slopes_t(ds, K)
                res[d] = sl[:, 0] - sl[:, 1]
        finally:
            m.motion_masking, m.len_min, m.len_max, m.n_steps = save
            m.p2_bypass = False
        return (res['noswap'] - res['swap']).astype(np.float32)


def evaluate(m, ev, od, ep, do_self):
    t0 = time.time()
    sav = ev.other(m)
    if do_self:
        sav['self_l1'] = np.float32(ev.self_l1(m))
        sav['c_self'] = ev.c_self(m)
    np.savez_compressed(os.path.join(od, 'eval', f'ep{ep:03d}.npz'), **sav)
    m.train()
    return time.time() - t0


def train(stage, cond, variant, ratio):
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    name = f'stage1_{cond}_{variant}' if stage == 1 else f'stage2_{cond}_r{ratio:g}'
    od = os.path.join(OUT, 'runs', name)
    assert not os.path.exists(os.path.join(od, 'done.json')), od
    for sub in ('model', 'eval'):
        os.makedirs(os.path.join(od, sub), exist_ok=True)
    m, exp_cfg = build(cond, variant == 'M' if stage == 1 else True, dev)
    if stage == 2:
        st = torch.load(os.path.join(OUT, 'runs', f'stage1_{cond}_M', 'model', f'{EP1:03d}.pth'), map_location='cpu')
        m.load_state_dict(st)
    set_trainable(m, stage)
    opt = optimizer(m, ratio if stage == 2 else 1.0)
    data = load_split('train')
    tl = loader.DataLoader(data, exp_cfg.train.batch_size, True, dev)
    ev = Evaluator(dev)
    runner = runners.MotionGenerationPredictionFeaturePredictionRunner()
    n_ep = EP1 if stage == 1 else EP2
    log = {'name': name, 'stage': stage, 'cond': cond, 'variant': variant, 'ratio': ratio, 'start': START[cond],
           'g_init_sha': None, 'epochs': []}
    import hashlib
    log['g_init_sha'] = hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for k, v in sorted(m.state_dict().items())
                                                if k.startswith('motion_generator_module'))).hexdigest()
    torch.save(m.state_dict(), os.path.join(od, 'model', '000.pth'))
    te = evaluate(m, ev, od, 0, True)
    log['eval_sec'] = [te]
    util.seed_all(SEED)
    m.train()
    for ep in range(1, n_ep + 1):
        t0 = time.time()
        acc = defaultdict(float)
        nb = 0
        for seq, _ in tl.load_sequence():
            loss, _ = runner.run_seq(m, seq, ep, exp_cfg)
            opt.zero_grad()
            sum(loss.values()).backward()
            opt.step()
            for k, v in loss.items():
                acc[k] += float(v)
            nb += 1
        rec = {'epoch': ep, 'sec': time.time() - t0, **{k: v / nb for k, v in acc.items()}}
        if ep % EVERY == 0:
            torch.save(m.state_dict(), os.path.join(od, 'model', f'{ep:03d}.pth'))
            rec['eval_sec'] = evaluate(m, ev, od, ep, stage == 2 or ep == n_ep)
        log['epochs'].append(rec)
        json.dump(log, open(os.path.join(od, 'log.json'), 'w'), indent=1)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    json.dump({'finished': time.strftime('%Y-%m-%d %H:%M')}, open(os.path.join(od, 'done.json'), 'w'))


def check():
    """G's initial weights equal across conditions; epoch-0 C_self reproduces round 21 (pair 0, ep200, 50 episodes);
    variant N at epoch 0 equals the plain stage-II model with the same weights; the evaluations run (shapes, times)."""
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    ev = Evaluator(dev, n=50)
    out = {}
    shas = {}
    import hashlib
    for cond in START:
        m, _ = build(cond, True, dev)
        shas[cond] = hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for k, v in sorted(m.state_dict().items())
                                             if k.startswith('motion_generator_module'))).hexdigest()
        t0 = time.time()
        cs = ev.c_self(m)
        z = np.load(os.path.join(RES, 'paper_extra', 'Round21', 'eval', f'{cond}_s0_ep200.npz'))
        ref = (z['noswap_V1_self_hold'] - z['swap_V1_self_hold'])[:50]
        out[f'{cond}|c_self_vs_round21_max_abs_diff'] = float(np.nanmax(np.abs(cs - ref)))
        out[f'{cond}|c_self_nan_same'] = bool(np.array_equal(np.isnan(cs), np.isnan(ref)))
        out[f'{cond}|c_self_sec'] = time.time() - t0
        t0 = time.time()
        o = ev.other(m)
        out[f'{cond}|other_sec'] = time.time() - t0
        out[f'{cond}|other_keys'] = sorted(o)
        t0 = time.time()
        out[f'{cond}|self_l1'] = ev.self_l1(m)
        out[f'{cond}|self_l1_sec'] = time.time() - t0
    out['g_init_equal_across_conditions'] = len(set(shas.values())) == 1
    # variant N at epoch 0 == plain stage-II model (identity adapter, no mask)
    mN, _ = build('p50', False, dev)
    exp_cfg, mcfg, _ = configs()
    plain = model_v12.SuperpositionNetworkMotionPleasureHoldOther(mcfg).to(dev)
    plain.load_state_dict({k: v for k, v in mN.state_dict().items() if not k.startswith(('p2_adapter', 'p2_mask'))})
    tl = loader.DataLoader(load_split('test', 20), 10, False, dev)
    diffs = []
    for mm in (mN, plain):
        mm.eval()
    with torch.no_grad():
        for seq, _ in tl.load_sequence():
            for x, truth, t in seq:
                if t == 0:
                    mN.init_state(10)
                    plain.init_state(10)
                p = 0 if t == 0 else 1
                diffs.append(float((mN(x, p, p)['self_vision'] - plain(x, p, p)['self_vision']).abs().max()))
    out['N_epoch0_vs_plain_stage2_max_abs_diff'] = max(diffs)
    os.makedirs(os.path.join(OUT, 'check'), exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'check', 'train_check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('stage1', 'stage2', 'check'))
    ap.add_argument('--cond', choices=tuple(START))
    ap.add_argument('--variant', choices=('M', 'N'), default='M')
    ap.add_argument('--ratio', type=float, default=0.1)
    a = ap.parse_args()
    if a.mode == 'check':
        check()
    else:
        train(1 if a.mode == 'stage1' else 2, a.cond, a.variant, a.ratio)
