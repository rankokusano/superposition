"""
Run the v11 figure scripts (pca_v11, pred_images_v11, check_convergence_v11) on v12 models without
changing any v11 code (docs/v12_experiment_log.md Sec.16).
The v11 scripts call model({'self_vision': x_t}, pm, pm) via eval_trial_v11.run(model, p_mask, h5path,
split, ...) and load models with eval_trial_v11.load_model. Here
  * load_model is replaced in the target module: v12 model classes are wrapped in MotionFeed, other
    classes (e.g. the Fig.5 Autoencoder) load as before;
  * run is replaced so that, before each pass over a dataset, the wrapper gets that split's
    self_motion array; the wrapper then injects x['self_motion'] for the current batch and step.
Usage: python analyze/v12/v11fig.py {pca|pred_images|convergence} <the v11 script's own arguments>
(use --sub ../v12/<dir> so outputs go under data/result/v12/).
"""

import os
import sys

import h5py
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
sys.path.insert(0, _HERE)
import eval_trial_v11  # noqa
from eval_trial_v12 import load_v12  # noqa
from model import model_v12  # noqa
from config_util import load_config  # noqa

_PI = os.path.dirname(os.path.dirname(_HERE))


class MotionFeed:
    def __init__(self, model):
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'motion', None)
        object.__setattr__(self, 'b0', 0)
        object.__setattr__(self, 'next0', 0)
        object.__setattr__(self, 't', 0)
        object.__setattr__(self, 'B', 0)

    def reset(self, motion):
        object.__setattr__(self, 'motion', motion)
        object.__setattr__(self, 'next0', 0)

    def init_state(self, batch_size):
        object.__setattr__(self, 'b0', self.next0)
        object.__setattr__(self, 'next0', self.next0 + batch_size)
        object.__setattr__(self, 'B', batch_size)
        object.__setattr__(self, 't', 0)
        self.model.init_state(batch_size)

    def __call__(self, x, pm_s, pm_o, **kw):
        x = dict(x)
        dev = x['self_vision'].device
        x['self_motion'] = torch.from_numpy(self.motion[self.b0:self.b0 + self.B, self.t]).float().to(dev)
        object.__setattr__(self, 't', self.t + 1)
        return self.model(x, pm_s, pm_o, **kw)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        setattr(self.model, name, value)


def load_model(exp, seed, epoch, device, ckpt_root=None):
    cfg = load_config(os.path.join(_PI, 'config', 'exp', exp + '.yml'))
    if hasattr(model_v12, cfg.model.name):
        m, p = load_v12(exp, seed, epoch, device)
        return MotionFeed(m), p
    if exp == 'v12_fig5_s0':
        seed = 0   # one Fig.5 Decoder for all v12 runs (the encoders are the same frozen v4 weights)
    return eval_trial_v11.load_model(exp, seed, epoch, device, ckpt_root)


def run(model, p_mask, h5path, split, n_ep, device, **kw):
    if isinstance(model, MotionFeed):
        model.reset(h5py.File(h5path, 'r')[split]['self_motion'][()])
    return eval_trial_v11.run(model, p_mask, h5path, split, n_ep, device, **kw)


def main():
    which = sys.argv.pop(1)
    if which == 'convergence':
        import check_convergence_v11 as mod
    else:
        mod = __import__({'pca': 'pca_v11', 'pred_images': 'pred_images_v11'}[which])
        mod.load_model = load_model
        mod.run = run
    sys.argv[0] = mod.__file__
    mod.main()


if __name__ == '__main__':
    main()
