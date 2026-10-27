"""
v15 (D) NS training (docs/v15_experiment_log.md Sec.33): train_v15.py / train_v12.py run unchanged, with the NS classes
(model/model_v15ns.py) registered in model_v12's namespace. Same arguments as train_v12.py.
"""
import os
import runpy
import sys

_PI_DIR = os.path.dirname(os.path.abspath(__file__))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import train_v15  # noqa: E402,F401  (registers the colour runner and the V15 classes; its runpy is under __main__)
from model import model_v12, model_v15ns  # noqa: E402
for _name in ('SuperpositionNetworkMotionPleasureHoldV15NS', 'SuperpositionNetworkMotionPleasureHoldOtherV15NS'):
    setattr(model_v12, _name, getattr(model_v15ns, _name))

if __name__ == '__main__':
    runpy.run_path(os.path.join(_PI_DIR, 'train_v12.py'), run_name='__main__')
