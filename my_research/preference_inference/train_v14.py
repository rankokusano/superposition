"""
v14 training entry (docs/v14_experiment_log.md): runs train_v12.py unchanged (runpy) after making the v14 model
classes visible in model.model_v12's namespace and routing util.gen_data_loader to exp/loader_v14 (which only
adds x['a1_liked_t'] when the dataset has that key; otherwise identical to exp/loader.py). No v12 file changes.
Run from the v11 worktree's my_research/preference_inference, with train_v12.py's arguments.
"""
import os
import runpy
import sys

_PI_DIR = os.path.dirname(os.path.abspath(__file__))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import util  # noqa
from model import model_v12, model_v14  # noqa
from exp import loader_v14  # noqa

for _name in ('SuperpositionNetworkMotionPleasureHoldOtherOneBit', 'SuperpositionNetworkMotionPleasureHoldSigned'):
    assert not hasattr(model_v12, _name)
    setattr(model_v12, _name, getattr(model_v14, _name))
util.gen_data_loader = loader_v14.gen_data_loader

if __name__ == '__main__':
    sys.argv[0] = os.path.join(_PI_DIR, 'train_v12.py')
    runpy.run_path(sys.argv[0], run_name='__main__')
