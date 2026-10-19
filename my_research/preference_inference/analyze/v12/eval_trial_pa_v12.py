"""
v12 trial (P-a) evaluation (docs/v12_experiment_log.md Sec.8): the paper's exp1 model (motion only,
never hidden) trained on v11 data. Common test, canonical vision mask 0.99, util.seed_all(0),
model.eval(), cudnn deterministic. Self-position tracking (all frames), self/other code alignment
(same formula as eval_trial_v12) and, for reference only, M1(c)/M1(d) on all frames.
Writes data/result/v12/trial/pa_eval_random_ep<e>.json.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v12 import COMMON, OUT, PI_DIR_CFG, alignment, batch_x, m1_all, tracking  # noqa
from eval_trial_v11 import RES  # noqa
import util  # noqa
from config_util import load_config  # noqa
import model as models  # noqa


def load_pa(exp, seed, epoch, dev):
    cfg = load_config(os.path.join(PI_DIR_CFG, 'exp', exp + '.yml'))
    mcfg = load_config(os.path.join(PI_DIR_CFG, 'model', cfg.model.name, cfg.model.config + '.yml'))
    m = getattr(models, cfg.model.name)(mcfg).to(dev)
    m.load_state_dict(torch.load(os.path.join(RES, exp, str(seed), 'model', f'{epoch:05d}.pth'), map_location='cpu')['model'])
    m.eval()
    return m, float(cfg.p_mask_vision)


@torch.no_grad()
def run(model, p_mask, dev, batch=50):
    g = h5py.File(COMMON, 'r')['test']
    N = g['self_vision'].shape[0]
    util.seed_all(0)
    H, E, P = [], [], []
    for b0 in range(0, N, batch):
        x, m = batch_x(g, b0, batch, dev)
        model.init_state(x.shape[0])
        h, e, p = [], [], []
        for t in range(100):
            pm = 0 if t == 0 else p_mask
            pr = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm)['self_vision']
            h.append(model.superposition_module.state['self'][0].cpu().numpy())
            p01 = ((pr + 1) / 2).clamp(0, 1)
            e.append((p01 - (x[:, t + 1] + 1) / 2).abs().mean((1, 2, 3)).cpu().numpy())
            p.append(p01.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16))
        H.append(np.stack(h, 1)); E.append(np.stack(e, 1)); P.append(np.stack(p, 1))
    h = np.concatenate(H)
    return {'h': h, 'err': np.concatenate(E), 'pred': np.concatenate(P), 'hidden': np.zeros(h.shape[:2], bool),
            'pos': g['self_position'][()], 'layout': g['layout'][()], 'src': g['a1_source'][()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, nargs='+', default=[60, 100, 150])
    args = ap.parse_args()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    fcf = h5py.File(os.path.join(RES, 'v11', 'cf', 'cf_views.h5'), 'r')
    cf, cfv = fcf['cf_vision'][()], fcf['valid'][()]
    for e in args.epochs:
        model, p_mask = load_pa('v12_trial_pa_random_mse', 0, e, dev)
        d = run(model, p_mask, dev)
        tr = tracking(d)
        r = {'exp': 'v12_trial_pa_random_mse', 'epoch': e, 'err_mean': float(d['err'].mean()),
             'tracking': {k: {'all': v['all'], 'err_median': v['err_median_visible']} for k, v in tr.items() if isinstance(v, dict)}}
        m1, _ = m1_all(d, cf, cfv)
        r['M1_all_frames_reference'] = m1['all']
        r['alignment'] = alignment(model)
        with open(os.path.join(OUT, f'pa_eval_random_ep{e}.json'), 'w') as f:
            json.dump(r, f, indent=1)
        print('pa', e, json.dumps({'track': r['tracking'], 'align': r['alignment']}), flush=True)


if __name__ == '__main__':
    main()
