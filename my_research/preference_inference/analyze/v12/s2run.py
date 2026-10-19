"""
Run the v11 stage-2 evaluation scripts on v12 stage-2 models without changing any v11 code
(docs/v12_experiment_log.md Sec.21). In the target module:
  * load_model: 'v11_s2_<st>_<c>_s<s>' is mapped to '<prefix>_<st>_<c>_s<s>' (default prefix v12_s2) and the
    v12 model is wrapped: self_motion is injected per batch/step (as analyze/v12/v11fig.py) and an
    8-dim p2_override (zero / P_A2 / position proxy / D6, as the v11 scripts pass it) becomes the v12
    process-2 input [motion 0, pleasure, flag 1];
  * the run functions get the dataset's self_motion before each pass;
  * output roots (OUT, OUT3, OUT4, OUTM, FIG) are moved to data/result/v12/<out>/... so v11 results are
    never touched.
Usage: python analyze/v12/s2run.py <eval_s2|t1t3|t4t6|m1d_other|figs_s2> [--s2prefix v12_s2] [--out stage2]
       <the v11 script's own arguments>
"""

import os
import sys

import h5py
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
sys.path.insert(0, _HERE)
import eval_trial_v11  # noqa
from eval_trial_v12 import RES, load_v12  # noqa
from v11fig import MotionFeed  # noqa


class S2Feed(MotionFeed):
    def __call__(self, x, pm_s, pm_o, p2_override=None, **kw):
        if p2_override is not None and p2_override.shape[-1] == 8:
            z = torch.zeros(p2_override.shape[0], 1, device=p2_override.device, dtype=p2_override.dtype)
            p2_override = torch.cat([z, z, p2_override, torch.ones_like(z)], 1)
        return super().__call__(x, pm_s, pm_o, p2_override=p2_override, **kw)


def main():
    which = sys.argv.pop(1)
    prefix, out = 'v12_s2', 'stage2'
    for flag in ('--s2prefix', '--out'):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            val = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            if flag == '--s2prefix':
                prefix = val
            else:
                out = val
    base = os.path.join(RES, 'v12', out)
    import eval_s2_v11
    eval_s2_v11.OUT = os.path.join(base, 'eval')
    mod = __import__({'eval_s2': 'eval_s2_v11', 't1t3': 't1t3_s2_v11', 't4t6': 't4t6_s2_v11',
                      'm1d_other': 'm1d_other_v11', 'figs_s2': 'figs_s2_v11'}[which])
    for name, sub in (('OUT', 'eval'), ('OUT3', 't1t3'), ('OUT4', 't4t6'), ('OUTM', 'm1d_other'), ('FIG', 'figs')):
        if hasattr(mod, name):
            setattr(mod, name, os.path.join(base, sub))
    if which == 'm1d_other':
        mod.CF = os.path.join(RES, 'v11', 'cf_other')   # counterfactual renders are model independent (read only)

    def load_model(exp, seed, epoch, device, ckpt_root=None):
        if exp.startswith('v11_s2_'):
            m, p = load_v12(prefix + exp[len('v11_s2'):], seed, epoch, device)
            return S2Feed(m), p
        return eval_trial_v11.load_model(exp, seed, epoch, device, ckpt_root)
    mod.load_model = load_model

    def wrap(fn, path_of):
        def w(model, *a, **k):
            if isinstance(model, S2Feed):
                path, split = path_of(*a)
                model.reset(h5py.File(path, 'r')[split]['self_motion'][()])
            return fn(model, *a, **k)
        return w
    for fname in ('run_q2', 'run', 'errs', 'g5a_est'):
        if hasattr(mod, fname) and callable(getattr(mod, fname)):
            if fname == 'run_q2':
                setattr(mod, fname, wrap(getattr(mod, fname), lambda path, split, *r: (path, split)))
            else:
                setattr(mod, fname, wrap(getattr(mod, fname), lambda d, *r: (d['path'], d['split'])))
    if which == 'figs_s2':
        mod.run_q2 = wrap(eval_s2_v11.run_q2, lambda path, split, *r: (path, split))
    sys.argv[0] = mod.__file__
    mod.main()


if __name__ == '__main__':
    main()
