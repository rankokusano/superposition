"""
v12 stage 2c (docs/v12_experiment_log.md Sec.45 G): forward-only check, NO training / no backward. With the stage-2c
config and loader, confirm that the network input view is self_vision_in (A-2 removed on hidden steps) and the
target is the original self_vision, and that the model runs and gives a finite loss. CPU, one batch.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI)
os.chdir(_PI)
import util  # noqa
from model import model_v12  # noqa
from exp import loader_v12  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', default='v12_s2c_green_mixed_s0')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    cfg = util.gen_exp_config(SimpleNamespace(exp_config=a.exp_config))
    model = getattr(model_v12, cfg.model.name)(util.gen_model_config(cfg))
    util.load_pretrain(model, cfg)
    model.eval()
    key = cfg.data.input_vision
    _, _, test = loader_v12.gen_data_loader(cfg.data.name, 10, False, torch.device('cpu'), key)
    raw = util.load_data(util.gen_data_filename(cfg.data.name), False)['test']
    seq, idx = next(iter(test.load_sequence()))
    H = raw['a2_hidden'][idx]
    sv_in = torch.from_numpy(util.scale_vision(util.transpose_vision(raw['self_vision_in'][idx])))
    sv = torch.from_numpy(util.scale_vision(util.transpose_vision(raw['self_vision'][idx])))
    model.init_state(len(idx))
    n_in = n_tg = n_diff = 0
    losses = []
    for x, y, t in seq:
        n_in += int(torch.equal(x['self_vision'], sv_in[:, t]))
        n_tg += int(torch.equal(y['self_vision'], sv[:, t + 1]))
        hid = H[:, t] == 1
        if hid.any():
            n_diff += int(not torch.equal(x['self_vision'][hid], sv[hid, t]))
        pm = 0 if t == 0 else 1
        out = model(x, pm, pm)
        losses.append(float(torch.nn.functional.l1_loss(out['self_vision'], y['self_vision'])))
    T = len(losses)
    print(f'{a.exp_config} ({cfg.data.name}, input {key}): input == self_vision_in at {n_in}/{T} steps; '
          f'target == self_vision[t+1] at {n_tg}/{T}; steps with a hidden episode whose input differs from '
          f'self_vision: {n_diff}; hidden fraction in batch {H[:, :T].mean():.3f}; l1 loss mean {np.mean(losses):.4f} '
          f'(finite {np.isfinite(losses).all()})')


if __name__ == '__main__':
    main()
