"""
v13-NS training (docs/v13_experiment_log.md Sec.19-20): train_v12.py run unchanged, with the NS model class
(model/model_v13.py) registered in model.model_v12's namespace so that train_v12's class lookup finds it.
Usage: same arguments as train_v12.py.
"""
import os
import runpy
import sys

_PI_DIR = os.path.dirname(os.path.abspath(__file__))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

from model import model_v12, model_v13  # noqa

model_v12.SuperpositionNetworkMotionPleasureHoldNS = model_v13.SuperpositionNetworkMotionPleasureHoldNS
runpy.run_path(os.path.join(_PI_DIR, 'train_v12.py'), run_name='__main__')
