"""
v15 stage 1 (docs/v15_experiment_log.md Sec.8): train_v12.py run unchanged, with the v15 colour runner
(exp/runner_v15.py) registered in exp.runner so a config can name `runner: PredictionFeaturePredictionColourRunner`.
Same arguments as train_v12.py. Run from the v15 worktree's my_research/preference_inference.
"""
import os
import runpy
import sys

_PI_DIR = os.path.dirname(os.path.abspath(__file__))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

from exp import runner, runner_v15  # noqa

runner.PredictionFeaturePredictionColourRunner = runner_v15.PredictionFeaturePredictionColourRunner

# v15 stage 2 (Sec.16): model classes looked up in model_v12 by train_v12.py
from model import model_v12, model_v15  # noqa: E402
for _name in ('SuperpositionNetworkMotionPleasureHoldV15', 'SuperpositionNetworkMotionPleasureHoldOtherV15'):
    setattr(model_v12, _name, getattr(model_v15, _name))

if __name__ == '__main__':
    runpy.run_path(os.path.join(_PI_DIR, 'train_v12.py'), run_name='__main__')
