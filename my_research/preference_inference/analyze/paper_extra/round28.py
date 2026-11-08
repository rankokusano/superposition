"""
Paper round 28 (paper/check/paper_instructions_round28.md; docs/paper_extra_log.md Sec.52; exploratory, outside the
pre-registration). Does the experience of one's own value hinder learning the other's preference-tied motion?
Sec.1 adds pairs 4-6 to round 27 Sec.2; Sec.2 removes the stage-I data from round 27 Sec.2 (p50, pairs 1-6). GPU
(GPU1-3). New code; existing models / data / results (incl. rounds 26 and 27) are only read.
Outputs: data/result/paper_extra/Round28/.

Terms T, W, Z, flows V / B, D_T, D_T,late (mean of epochs 40, 45, 50): as rounds 26-27 (round26.py, round27.py).

Modes
  stage_a  Sec.1 step 1 (pairs 4-6 = seeds 3-5): round 26 Sec.2 B exactly (round26.train2 with flow B, called as is;
           only its output root is moved): adapter only, T instead of G, ranges 2-5 masked after the adapter, 50 epochs.
           -> Round28/stage_a/runs/s2_<cond>_B_s<pair>/ (same layout as Round26/runs)
  train    Sec.1 step 2 (--mix 1) and Sec.2 (--mix 0): round 27 Sec.2 (round27.train) with the start adapter
           taken from round 26 Sec.2 B ep50 for pairs 1-3 (Round26/runs/s2_<cond>_B_s<pair>/model/050.pth) and from
           stage_a ep50 for pairs 4-6 (Round28/stage_a/runs/s2_<cond>_B_s<pair>/model/050.pth). Trainable: adapter
           (lr 1e-3) + shared LSTM (superposition_module, lr 1e-4 = 1/10), AdamW wd 0.01 (bias 0), 50 epochs, 300
           iterations per epoch, seed R25.SEED before epoch 1.
             --mix 1: every iteration = 10 episodes of paper_r25_swap50 train in flow B (stage-II loss L_II) + 10
                      episodes of the condition's own stage-I training data (stage-I flow and l1 loss L_I; sr50 with
                      its episode signs); loss = 0.5 (L_II + L_I). Identical to round27.train (pairs 4-6 only here;
                      pairs 1-3 are round 27's runs). Evaluation every 5 epochs incl. 0 (D_T, C_self, L1 of A-1's
                      view V / B) and the Sec.1 window sums at epochs 0 and 50 (r25 train / test, flow B).
                      -> Round28/runs/mix_<cond>_s<pair>/
             --mix 0: p50 only; every iteration = the same 10 r25 episodes in flow B; loss = 0.5 L_II (the stage-II
                      term keeps the weight it has in the mixed loss; only L_I is removed). Same evaluation every 5
                      epochs (D_T, C_self, L1 V / B); no window sums. -> Round28/runs/nomix_p50_s<pair>/
  check    (a) stage_a reproduces round 26: round26.train2('p50', 'B', pair 1) for 5 epochs into Round28/check/repro;
               ep000 / ep005 evaluations and adapters vs Round26/runs/s2_p50_B_s0.
           (b) the stage-I predictors of pairs 4-6 (none / p50 / sr50) load into the round-27 model; the epoch-0
               D_T of the mix-0 start for pair 1 equals round 27 s2_p50_s0 ep000 (first 50 test episodes).
           (c) mix 0: gradient only to the adapter and the shared LSTM; seconds per iteration.
           -> Round28/check/check_a.json, check.json
Usage: python analyze/paper_extra/round28.py stage_a --cond p50 --pair 3 | train --cond p50 --pair 3 --mix 1 |
       check a | check bc
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round27 as R27  # noqa
import round26 as R26  # noqa
import round25_train as R25  # noqa
import util  # noqa
from exp import runner as runners, loader  # noqa

RES = R25.RES
OUT = os.path.join(RES, 'paper_extra', 'Round28')
STAGE_A = os.path.join(OUT, 'stage_a')


def start_adapter(cond, pair):
    if pair < 3:
        return R27.r26_adapter(cond, pair)
    d = os.path.join(STAGE_A, 'runs', f's2_{cond}_B_s{pair}')
    assert os.path.exists(os.path.join(d, 'done.json')), d
    return os.path.join(d, 'model', '050.pth')


def stage_a(cond, pair):
    assert pair >= 3
    R26.OUT = STAGE_A
    R26.train2(cond, 'B', pair)


def train(cond, pair, mix, n_ep=50, every=5):
    assert mix or cond == 'p50'
    assert not (mix and pair < 3), 'mixed pairs 1-3 are round 27 runs'
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    name = f"{'mix' if mix else 'nomix'}_{cond}_s{pair}"
    od = os.path.join(OUT, 'runs', name)
    assert not os.path.exists(od), od
    for sub in ('model', 'eval'):
        os.makedirs(os.path.join(od, sub))
    exp_cfg, _, _ = R25.configs()
    adp = start_adapter(cond, pair)
    m = R27.build27(cond, pair, dev, adp)
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith(('p2_adapter', 'superposition_module'))
    opt = R25.optimizer(m, 0.1)
    d, sw, s, lk = R26.load25('train')
    tl2 = loader.DataLoader(d, exp_cfg.train.batch_size, True, dev)
    run2 = runners.MotionGenerationPredictionFeaturePredictionRunner()
    if mix:
        cfg1 = R27.s1_cfg(cond)
        d1, liked1 = R27.load_s1(cond, 'train')
        tl1 = loader.DataLoader(d1, cfg1.train.batch_size, True, dev)
        run1 = runners.PredictionFeaturePredictionRunner()
    ev = R27.Eval27(dev)
    log = {'name': name, 'cond': cond, 'pair': pair, 'mix': bool(mix), 'start_predictor': R26.s1_path(cond, pair),
           'start_adapter': adp, 'stage1_data': R27.DS1[cond] if mix else None,
           'stage1_signs': R27.SIGNS if (mix and liked1 is not None) else None,
           'stage1_cfg': R27.S1CFG[cond] if mix else None,
           'trainable': sorted(n for n, p in m.named_parameters() if p.requires_grad), 'epochs': []}

    def evaluate(ep):
        t0 = time.time()
        sav = ev.e2.other(m)
        sav['c_self'] = ev.e25.c_self(m)
        sav['l1_T_V'] = np.float32(ev.l1_T(m, 'V'))
        sav['l1_T_B'] = np.float32(ev.l1_T(m, 'B'))
        np.savez_compressed(os.path.join(od, 'eval', f'ep{ep:03d}.npz'), **sav)
        if mix and ep in (0, n_ep):
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
        it = zip(tl2.load_sequence(), tl1.load_sequence()) if mix else ((b, None) for b in tl2.load_sequence())
        for (seq2, b2), x1 in it:
            l2 = R26.seq_T(m, seq2, ep, exp_cfg, run2, 'B', torch.from_numpy(s[b2]).to(dev), R26.sign_of(lk[b2], dev))
            if mix:
                seq1, b1 = x1
                l1 = R27.seq_stage1(m, seq1, ep, cfg1, run1, None if liked1 is None else liked1[b1])
                loss = 0.5 * (sum(l2.values()) + sum(l1.values()))
            else:
                l1 = {}
                loss = 0.5 * sum(l2.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in l2.items():
                acc[k] += float(v)
            for k, v in l1.items():
                acc['s1_' + k] += float(v)
            nb += 1
        rec = {'epoch': ep, 'sec': time.time() - t0, 'iters': nb, **{k: v / nb for k, v in acc.items()}}
        if ep % every == 0:
            rec['eval_sec'] = evaluate(ep)
        log['epochs'].append(rec)
        json.dump(log, open(os.path.join(od, 'log.json'), 'w'), indent=1)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    json.dump({'finished': time.strftime('%Y-%m-%d %H:%M')}, open(os.path.join(od, 'done.json'), 'w'))


# ---------------------------------------------------------------- check
def check_a():
    """round26.train2 (p50, B, pair 1) for 5 epochs into Round28/check/repro vs Round26/runs/s2_p50_B_s0."""
    root = os.path.join(OUT, 'check', 'repro')
    R26.OUT = root
    t0 = time.time()
    R26.train2('p50', 'B', 0, n_ep=5)
    out = {'sec': time.time() - t0}
    new = os.path.join(root, 'runs', 's2_p50_B_s0')
    old = os.path.join(R26.RES, 'paper_extra', 'Round26', 'runs', 's2_p50_B_s0')
    for ep in (0, 5):
        a = np.load(os.path.join(new, 'eval', f'ep{ep:03d}.npz'))
        b = np.load(os.path.join(old, 'eval', f'ep{ep:03d}.npz'))
        out[f'ep{ep}_eval_max_abs_diff'] = {k: float(np.nanmax(np.abs(a[k].astype(np.float64) - b[k])))
                                            for k in a.files if a[k].dtype.kind == 'f'}
        wa = torch.load(os.path.join(new, 'model', f'{ep:03d}.pth'), map_location='cpu')
        wb = torch.load(os.path.join(old, 'model', f'{ep:03d}.pth'), map_location='cpu')
        out[f'ep{ep}_adapter_max_abs_diff'] = max(float((wa[k] - wb[k]).abs().max()) for k in wa)
    la = json.load(open(os.path.join(new, 'log.json')))['epochs']
    lb = json.load(open(os.path.join(old, 'log.json')))['epochs'][:5]
    out['train_loss_epochs_1_5'] = [[{k: v for k, v in x.items() if k not in ('sec', 'eval_sec', 'epoch')} for x in la],
                                    [{k: v for k, v in x.items() if k not in ('sec', 'eval_sec', 'epoch')} for x in lb]]
    json.dump(out, open(os.path.join(OUT, 'check', 'check_a.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


def check_bc():
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    out = {}
    # (b) pairs 4-6 load; epoch-0 D_T of the mix-0 start (p50 pair 1) == round 27 s2_p50_s0 ep000
    for cond in ('none', 'p50', 'sr50'):
        for pair in (3, 4, 5):
            R27.build27(cond, pair, torch.device('cpu'))
            out[f'b_load_{cond}_s{pair}'] = R26.s1_path(cond, pair)
    m = R27.build27('p50', 0, dev, start_adapter('p50', 0))
    e2 = R26.Eval2(dev, n=50).other(m)
    zr = np.load(os.path.join(R27.OUT, 'runs', 's2_p50_s0', 'eval', 'ep000.npz'))
    out['b_ep0_vs_round27_ep000_max_abs_diff'] = {k: float(np.nanmax(np.abs(e2[k] - zr[k][:50])))
                                                  for k in ('S_T_delta', 'O_T_delta', 'S_Z_delta', 'O_Z_delta')}
    # (c) mix 0: gradients and timing
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith(('p2_adapter', 'superposition_module'))
    opt = R25.optimizer(m, 0.1)
    out['c_param_groups_lr'] = sorted({(g_['lr'], g_['weight_decay'], len(g_['params'])) for g_ in opt.param_groups})
    exp_cfg, _, _ = R25.configs()
    dd, _, ss, lkk = R26.load25('test', 100)
    tl2 = loader.DataLoader({k: dd[k] for k in R26.KEYS}, 10, True, dev)
    run2 = runners.MotionGenerationPredictionFeaturePredictionRunner()
    m.train()
    ts = []
    for i, (seq2, b2) in enumerate(tl2.load_sequence()):
        t0 = time.time()
        l2 = R26.seq_T(m, seq2, 1, exp_cfg, run2, 'B', torch.from_numpy(ss[b2]).to(dev), R26.sign_of(lkk[b2], dev))
        opt.zero_grad()
        (0.5 * sum(l2.values())).backward()
        g = sorted({n.split('.')[0] for n, p in m.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0})
        opt.step()
        ts.append(time.time() - t0)
        if i == 2:
            break
    out['c_modules_with_grad'] = g
    out['c_sec_per_iter'] = float(np.median(ts))
    json.dump(out, open(os.path.join(OUT, 'check', 'check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('stage_a', 'train', 'check'))
    ap.add_argument('part', nargs='?', default='bc')
    ap.add_argument('--cond', choices=('none', 'p50', 'sr50'))
    ap.add_argument('--pair', type=int, default=0)
    ap.add_argument('--mix', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'check'), exist_ok=True)
    if a.mode == 'stage_a':
        stage_a(a.cond, a.pair)
    elif a.mode == 'train':
        train(a.cond, a.pair, a.mix)
    elif a.part == 'a':
        check_a()
    else:
        check_bc()
