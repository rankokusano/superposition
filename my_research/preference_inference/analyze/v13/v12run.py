"""
Run the v12 stage-1 evaluation scripts on the v13 pairs without changing any v12 code
(docs/v13_experiment_log.md Sec.3.2 C). A v13 pair s is (no experience: the v12 run v12_s1_random_*,
reused, Sec.1; experienced 80%: v13_s1_mixed_*). The v12 scripts build model names as
exp_fmt.format(cond); here they are called with exp_fmt 'v13_s1_{}_mse' / 'v13_s1_{}_l1_s<s>' and the
model loader maps the name before reading config and weights:
    v13_s1_random_*  ->  v12_s1_random_*     (read only; nothing is written under the v12 runs)
    v13_s1_mixed_*   ->  v13_s1_mixed_*
For the reference runs (Sec.3.3 splits, Sec.3.4 comparison) the prefix 'v13ref_s1_' maps to v12's own
pair: v13ref_s1_random_* -> v12_s1_random_*, v13ref_s1_mixed_* -> v12_s1_mixed_* (the 50% runs).
Usage (inside the container, from my_research/preference_inference; outputs via --out/--sub ../v13/...):
    python analyze/v13/v12run.py trial  <eval_trial_v12.py arguments>
    python analyze/v13/v12run.py g5     <eval_g5_v13.py arguments>
    python analyze/v13/v12run.py d6     <eval_d6_v12.py arguments>
    python analyze/v13/v12run.py track  <track_split_v13.py arguments>
    python analyze/v13/v12run.py h2     --runs random:<s>:200 mixed:<s>:200 --sub ../v13/main/h2_check_s<s>
    python analyze/v13/v12run.py convergence --stage mse|l1 --exp_fmt v13_s1_{c}_... --sub ../v13/main/convergence
    python analyze/v13/v12run.py pca|pred_images  <the v11 script's arguments, exp_fmt v13_s1_...>
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, _PI)


def map_exp(exp):
    for src, dst in (('v13_s1_random_', 'v12_s1_random_'), ('v13ref_s1_random_', 'v12_s1_random_'),
                     ('v13ref_s1_mixed_', 'v12_s1_mixed_')):
        if exp.startswith(src):
            return dst + exp[len(src):]
    return exp


def wrap(fn):
    return lambda exp, *a, **k: fn(map_exp(exp), *a, **k)


def main():
    which = sys.argv.pop(1)
    if which in ('trial', 'g5', 'd6', 'track'):
        mod = __import__({'trial': 'eval_trial_v12', 'g5': 'eval_g5_v13', 'd6': 'eval_d6_v12',
                          'track': 'track_split_v13'}[which])
        mod.load_v12 = wrap(mod.load_v12)
    elif which == 'convergence':   # reads the training logs of data/result/<exp>
        import check_convergence_v11 as mod
        mod.load = wrap(mod.load)
    else:
        import v11fig
        v11fig.load_model = wrap(v11fig.load_model)
        if which == 'h2':   # h2_check_v11 reads 'v11_s1_<c>_mse'
            import h2_check_v11 as mod
            names = {'v11_s1_random_mse': 'v12_s1_random_mse', 'v11_s1_mixed_mse': 'v13_s1_mixed_mse'}
            mod.load_model = lambda exp, *a, **k: v11fig.load_model(names[exp], *a, **k)
        else:
            mod = __import__({'pca': 'pca_v11', 'pred_images': 'pred_images_v11'}[which])
            mod.load_model = v11fig.load_model
        mod.run = v11fig.run
    sys.argv[0] = mod.__file__
    mod.main()


if __name__ == '__main__':
    main()
