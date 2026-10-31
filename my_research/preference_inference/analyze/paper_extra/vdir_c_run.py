"""
Paper round 13 rev2, Sec.1.6 (docs/paper_extra_log.md Sec.37.3; exploratory, stage-A extension, outside the
pre-registration): the matching candidates (c+, c-, c0, ref_red, ref_green) of eval_inverse_v14.py (unchanged) on
E1's 50% sign-randomised runs (paper_extra/E1_signrand50/s1_signrand50_*), as c_run.py's 'inverse' did for the 80%
sign-randomised runs. Specs: csr50:signrand50:<s> (l1 stage), csr50mse:signrand50:<s> (mse stage).
Usage (inside the container, from my_research/preference_inference):
    python analyze/paper_extra/vdir_c_run.py --models csr50mse:signrand50:0 --sets pref \
        --out ../paper_extra/Vdir_B/c_sr50/inverse
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import c_run  # noqa  (registers the signed-pleasure class in model_v12)
import eval_inverse_v14 as mod  # noqa
import eval_trial_v12  # noqa

NAMES = {'csr50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s{s}', 'csr50mse': 'paper_extra/E1_signrand50/s1_signrand50_mse'}
_load = mod.load


def load(spec, epoch, dev):
    fam, c, s = spec.split(':')
    if fam in NAMES:
        m, _ = eval_trial_v12.load_v12(NAMES[fam].format(s=s), int(s), epoch, dev)
        m.motion_masking = False   # stage-2 protocol, as eval_inverse_v14.load / c_run.py inverse
        return m, 11
    return _load(spec, epoch, dev)


mod.load = load

if __name__ == '__main__':
    sys.argv[0] = mod.__file__
    mod.main()
