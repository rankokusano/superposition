"""
Paper round 11 E1 evaluation (docs/paper_extra_log.md Sec.35.4-35.5): c_run.py's evaluations (unchanged) on the 50%
sign-randomised runs (paper_extra/E1_signrand50/s1_signrand50_*) and, for d_P, on the 50% runs (v12_s1_mixed_mse),
with the outputs under data/result/paper_extra/E1_signrand50/eval/ instead of C_signrand/eval/.
Usage (inside the container): python analyze/paper_extra/e1_run.py <trial|g5|decomp|g3flip> <c_run.py arguments>
    conds: signrand50 (all), mixed50 (g3flip only)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import c_run  # noqa

c_run.OUT = os.path.join(c_run.RES, 'paper_extra', 'E1_signrand50', 'eval')
c_run.MSE.update({'signrand50': 'paper_extra/E1_signrand50/s1_signrand50_mse', 'mixed50': 'v12_s1_mixed_mse'})
c_run.L1.update({'signrand50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s{s}'})

if __name__ == '__main__':
    assert sys.argv[1] in ('trial', 'g5', 'decomp', 'g3flip'), sys.argv[1]
    c_run.main()
