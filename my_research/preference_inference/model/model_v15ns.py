"""
v15 (D) (docs/v15_experiment_log.md Sec.33): the v15 stage-2 models without self/other sharing of the superposition LSTM
(the v13-NS idea, model/model_v13.py, applied to the v15 classes). Process-1 runs `lstm`, process-2 its own `lstm_other`,
a deep copy of `lstm` made after the whole model is built (no random numbers drawn): with the same seed the initial
weights, batch order, hiding schedule and dropout masks are those of the shared v15 run, so NS seed s pairs with shared
seed s. Freezing by name ('superposition_module') covers both LSTMs, so phase 1 trains both and phase 2 trains G only,
as in the shared runs. share_lns (frozen) and the feature-prediction module (detached input) stay shared.
"""

from .model_v13 import SplitSuperpositionModule
from .model_v15 import SuperpositionNetworkMotionPleasureHoldOtherV15, SuperpositionNetworkMotionPleasureHoldV15


class _Split(object):
    def _split(self):
        old = self.superposition_module
        self.superposition_module = SplitSuperpositionModule(old)
        self.rnn_modules = [self.superposition_module if m is old else m for m in self.rnn_modules]


class SuperpositionNetworkMotionPleasureHoldV15NS(_Split, SuperpositionNetworkMotionPleasureHoldV15):
    def __init__(self, config):
        super().__init__(config)
        self._split()


class SuperpositionNetworkMotionPleasureHoldOtherV15NS(_Split, SuperpositionNetworkMotionPleasureHoldOtherV15):
    def __init__(self, config):
        super().__init__(config)
        self._split()
