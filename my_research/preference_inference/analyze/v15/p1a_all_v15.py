"""
v15 Sec.46 check (Kusano-san, 2026-10-29): the pre-registered phase-1 precondition P1-a (Sec.16.3: usual run, h1 -> A-1 position,
R^2 >= 0.80 on motion-visible frames) for every phase-1 model used in phase 2, with exactly the measure of eval_s2_v15.py
phase1_tests (same v15_s1_p80_half_3000 test, same usual run and hide schedule, same 2-fold Ridge readout). CPU only; reads the
saved phase-1 checkpoints (ep150) and writes nothing else. Also P1-b (h2 -> A-2 position) and P1-a on all frames (descriptive).
The R1-R2b phase-1 checkpoints (analyze/v15/make_r12_v15.py, make_r1b_v15.py, make_r2b_v15.py) have no config of their own;
they are loaded into the NS class of v15_s2p1_noexp_v15i_ns_mse (strict).

    python analyze/v15/p1a_all_v15.py --tag exp_v4_mse --seed 1   # -> data/result/v15/stage2/p1a/p1a_<tag>_s<seed>.json
    --data v15_s1_rankY_p80_half_1000t (descriptive only): the same measure on the rank-Y test (-> ..._<data>.json)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
import util  # noqa
from model import model_v15, model_v15ns  # noqa
for _name in ('SuperpositionNetworkMotionPleasureHoldV15NS', 'SuperpositionNetworkMotionPleasureHoldOtherV15NS'):
    setattr(model_v15, _name, getattr(model_v15ns, _name))
from analyze.v15 import eval_s2_v15 as E  # noqa

CFG_OF = {'noexp_v15i_r1': 'noexp_v15i_ns_mse', 'noexp_v15i_r1b': 'noexp_v15i_ns_mse',
          'noexp_v15i_r2': 'noexp_v15i_ns_mse', 'noexp_v15i_r2b': 'noexp_v15i_ns_mse'}


def load1(tag, seed, dev, epoch=150):
    cfg = util.load_config(os.path.join(_PI_DIR, 'config', 'exp', f'v15_s2p1_{CFG_OF.get(tag, tag)}.yml'))
    mcfg = util.load_config(os.path.join(_PI_DIR, 'config', 'model', cfg.model.name, cfg.model.config + '.yml'))
    m = getattr(model_v15, cfg.model.name)(mcfg).to(dev)
    p = os.path.join(E.RES, f'v15_s2p1_{tag}', str(seed), 'model', f'{epoch:05d}.pth')
    m.load_state_dict(torch.load(p, map_location='cpu')['model'])
    return m.eval(), cfg.model.name, cfg.model.config, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data', default='v15_s1_p80_half_3000', help='test set (default: the pre-registered one)')
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = torch.device('cpu')
    m1, cls, mc, path = load1(a.tag, a.seed, dev)
    d = E.read(a.data, 'test', ['self_vision', 'self_motion', 'self_position', 'other_position'])
    N = d['self_vision'].shape[0]
    fo = E.folds(N)
    hid = E.schedule(m1, N)
    usual = E.run(m1, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 0.99),
                  hidden=lambda t: hid[:, t])
    vis = ~hid
    r = {'tag': a.tag, 'seed': a.seed, 'data': a.data, 'checkpoint': os.path.relpath(path, _PI_DIR), 'class': cls, 'model_config': mc,
         'device': 'cpu', 'n_episodes': int(N), 'frac_motion_visible': float(vis.mean()),
         'P1a_r2_self_visible': E.r2_cv(usual['h1'], d['self_position'], vis, fo),
         'P1a_r2_self_all': E.r2_cv(usual['h1'], d['self_position'], np.ones_like(vis), fo),
         'P1b_r2_other': E.r2_cv(usual['h2'], d['other_position'], np.ones_like(vis), fo)}
    out = os.path.join(E.OUT, 'p1a')
    os.makedirs(out, exist_ok=True)
    sfx = '' if a.data == 'v15_s1_p80_half_3000' else f'_{a.data}'
    json.dump(r, open(os.path.join(out, f'p1a_{a.tag}_s{a.seed}{sfx}.json'), 'w'), indent=1)
    print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
