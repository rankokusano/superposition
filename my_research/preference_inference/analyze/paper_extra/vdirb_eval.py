"""
Paper round 13 rev2 (V-dir stage B, docs/paper_extra_log.md Sec.37; pre-registered): the matching candidates
(c+, c-, c0, ref_red, ref_green; eval_inverse_v14.py unchanged) on the new stage-B set paper_vdir_b_pref
(registered here as 'vdirb'). Model specs as eval_inverse_v14 (v12mse|v12|v13mse|v13|v14cmse|v14c:<cond>:<s>),
plus csrmse|csr:signrand:<s> (80% sign-randomised, as c_run.py inverse) and csr50mse|csr50:signrand50:<s>
(50% sign-randomised, as vdir_c_run.py).
Usage (inside the container): python analyze/paper_extra/vdirb_eval.py --models <spec> ... --sets vdirb \
    --out ../paper_extra/Vdir_B/eval/inverse
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vdir_c_run as VC  # noqa  (c_run class registration, csr50 loader)
import eval_s2_v11  # noqa

eval_s2_v11.SETS['vdirb'] = ('paper_vdir_b_pref', 'test')
VC.NAMES.update({'csr': VC.c_run.L1['signrand'], 'csrmse': VC.c_run.MSE['signrand']})

if __name__ == '__main__':
    sys.argv[0] = VC.mod.__file__
    VC.mod.main()
