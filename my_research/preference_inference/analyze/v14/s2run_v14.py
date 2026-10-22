"""
Run the v12 stage-2 evaluation wrapper (analyze/v12/s2run.py, which runs the v11 scripts) on v14-B models without
changing any v11/v12 code (docs/v14_experiment_log.md Sec.4, 15): the v14 model classes are made visible in
model.model_v12's namespace so load_v12 can build them, then s2run.main() runs with the given arguments.
Outputs: pass --out ../v14/<dir> so they go to data/result/v14/<dir>/ (s2run joins data/result/v12/<out>).
Usage: python analyze/v14/s2run_v14.py <eval_s2|t1t3|...> --s2prefix v14_s2u --out ../v14/stage2u <script args>
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, _PI)
from model import model_v12, model_v14  # noqa

for _name in ('SuperpositionNetworkMotionPleasureHoldOtherOneBit', 'SuperpositionNetworkMotionPleasureHoldSigned'):
    if not hasattr(model_v12, _name):
        setattr(model_v12, _name, getattr(model_v14, _name))

import s2run  # noqa

if __name__ == '__main__':
    assert '--out' in sys.argv and sys.argv[sys.argv.index('--out') + 1].startswith('../v14/'), 'outputs must go to data/result/v14/'
    s2run.main()
