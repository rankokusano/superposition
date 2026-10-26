"""
Paper extra B (docs/paper_extra_log.md Sec.1.3, 3): run the existing evaluation scripts on the "copy" models of
v13-NS without changing them. A model name 'v13nscp_s1_<c>_<stage>' loads the v13-NS checkpoint
'v13ns_s1_<c>_<stage>' and then, in memory only, sets superposition_module.lstm_other (weight_ih, weight_hh, bias_ih,
bias_hh) to the values of superposition_module.lstm (process-1's). Every load asserts that lstm_other == lstm and
that the other 44 tensors equal the checkpoint. Checkpoint files are never written.
'v13nscpp_s1_<c>_<stage>' (log Sec.13) copies only the pleasure columns of the input weights,
lstm_other.weight_ih[:, 66:74] <- lstm.weight_ih[:, 66:74] (input = [vision 64, motion 2, pleasure 8, flag 1]);
the recurrent weights, the biases and the other input columns stay process-2's own.
Usage (inside the container, from my_research/preference_inference; outputs under data/result/paper_extra/B_copy):
    python analyze/paper_extra/b_copy_run.py trial   <eval_trial_v12.py arguments>     (G1-G3, self error)
    python analyze/paper_extra/b_copy_run.py g5      <eval_g5_v13.py arguments>        (G5a, G5b)
    python analyze/paper_extra/b_copy_run.py h2      --runs random:<s>:200 mixed:<s>:200 --sub ../paper_extra/B_copy/...
                                                     (h2_check_v11 with v11_s1_<c>_mse -> v13nscp_s1_<c>_mse)
    python analyze/paper_extra/b_copy_run.py decomp  <g5a_decomp_v13.py arguments>     (sign-following part;
                                                     random / mixed80 = v13nscp runs, output under B_copy/decomp)
    python analyze/paper_extra/b_copy_run.py inverse <eval_inverse_v14.py arguments>   (model specs v13nscp:<c>:<s>,
                                                     v13nscpmse:<c>:<s>)
    python analyze/paper_extra/b_copy_run.py d6      <a_cross_eval.py arguments>       (conds nscp_mixed, ...)
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_PI, 'analyze', 'v13'), os.path.join(_PI, 'analyze', 'v12'),
          os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)

from model import model_v12, model_v13  # noqa
model_v12.SuperpositionNetworkMotionPleasureHoldNS = model_v13.SuperpositionNetworkMotionPleasureHoldNS
import eval_trial_v12  # noqa

PREFIX, PREFIX_P, NS_PREFIX = 'v13nscp_', 'v13nscpp_', 'v13ns_'
P_COLS = slice(66, 74)   # pleasure columns of the 75-dim LSTM input: vision 0-63, motion 64-65, pleasure 66-73, flag 74
LSTM_KEYS = ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')
_load_v12 = eval_trial_v12.load_v12


def copy_lstm(m):
    before = {k: v.clone() for k, v in m.state_dict().items()}
    sm = m.superposition_module
    sm.lstm_other.load_state_dict(sm.lstm.state_dict())
    after = m.state_dict()
    for k in LSTM_KEYS:
        assert torch.equal(after[f'superposition_module.lstm_other.{k}'], after[f'superposition_module.lstm.{k}'])
    changed = [k for k in after if not torch.equal(after[k], before[k])]
    assert all(k.startswith('superposition_module.lstm_other.') for k in changed), changed
    return m


def copy_pleasure_cols(m):
    before = {k: v.clone() for k, v in m.state_dict().items()}
    sm = m.superposition_module
    with torch.no_grad():
        sm.lstm_other.weight_ih[:, P_COLS] = sm.lstm.weight_ih[:, P_COLS]
    after = m.state_dict()
    wi = 'superposition_module.lstm_other.weight_ih'
    assert torch.equal(after[wi][:, P_COLS], after['superposition_module.lstm.weight_ih'][:, P_COLS])
    keep = torch.ones(after[wi].shape[1], dtype=torch.bool)
    keep[P_COLS] = False
    assert torch.equal(after[wi][:, keep], before[wi][:, keep])
    changed = [k for k in after if not torch.equal(after[k], before[k])]
    assert set(changed) <= {wi}, changed
    return m


def load_v12(exp, seed, epoch, dev):
    if exp.startswith(PREFIX_P):
        m, p = _load_v12(NS_PREFIX + exp[len(PREFIX_P):], seed, epoch, dev)
        return copy_pleasure_cols(m), p
    if exp.startswith(PREFIX):
        m, p = _load_v12(NS_PREFIX + exp[len(PREFIX):], seed, epoch, dev)
        return copy_lstm(m), p
    return _load_v12(exp, seed, epoch, dev)


eval_trial_v12.load_v12 = load_v12   # before the evaluation modules import it


def main():
    which = sys.argv.pop(1)
    if which == 'trial':
        mod = eval_trial_v12
    elif which == 'g5':
        import eval_g5_v13 as mod
    elif which == 'h2':
        import v11fig
        import h2_check_v11 as mod
        pf = os.environ.get('B_PREFIX', 'v13nscp')
        names = {'v11_s1_random_mse': f'{pf}_s1_random_mse', 'v11_s1_mixed_mse': f'{pf}_s1_mixed_mse'}
        mod.load_model = lambda exp, *a, **k: v11fig.load_model(names[exp], *a, **k)
        mod.run = v11fig.run
    elif which == 'decomp':
        import g5a_decomp_v13 as mod
        pf = os.environ.get('B_PREFIX', 'v13nscp')
        mod.EXPS = {'random': f'{pf}_s1_random_l1_s{{s}}', 'mixed80': f'{pf}_s1_mixed_l1_s{{s}}'}
        mod.RES = os.path.join(eval_trial_v12.RES, 'paper_extra', os.environ.get('B_OUT', 'B_copy'), 'decomp')
    elif which == 'inverse':
        import eval_inverse_v14 as mod
        _load = mod.load

        def load(spec, epoch, dev):
            fam, c, s = spec.split(':')
            if fam in ('v13nscpp', 'v13nscppmse'):
                m, w = _load(spec.replace('v13nscpp', 'v13ns', 1), epoch, dev)
                return copy_pleasure_cols(m), w
            if fam in ('v13nscp', 'v13nscpmse'):
                m, w = _load(spec.replace('v13nscp', 'v13ns', 1), epoch, dev)
                return copy_lstm(m), w
            return _load(spec, epoch, dev)
        mod.load = load
    elif which == 'd6':
        import a_cross_eval as mod
        mod.EXPS.update({'ns_mixed': 'v13ns_s1_mixed_l1_s{s}', 'ns_random': 'v13ns_s1_random_l1_s{s}',
                         'nscp_mixed': 'v13nscp_s1_mixed_l1_s{s}', 'nscp_random': 'v13nscp_s1_random_l1_s{s}',
                         'nscpp_mixed': 'v13nscpp_s1_mixed_l1_s{s}', 'nscpp_random': 'v13nscpp_s1_random_l1_s{s}'})
        mod.OUT = os.path.join(eval_trial_v12.RES, 'paper_extra', os.environ.get('B_OUT', 'B_copy'), 'd6')
    else:
        raise SystemExit(f'unknown {which}')
    for m in (sys.modules.get('eval_g5_v13'), sys.modules.get('g5a_decomp_v13'), sys.modules.get('eval_inverse_v14'),
              sys.modules.get('a_cross_eval'), sys.modules.get('v11fig')):
        if m is not None and hasattr(m, 'load_v12'):
            m.load_v12 = load_v12
    sys.argv[0] = mod.__file__
    mod.main()


if __name__ == '__main__':
    main()
