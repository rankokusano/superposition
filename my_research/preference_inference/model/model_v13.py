"""
v13-NS (docs/v13_experiment_log.md Sec.19-20): the v13 stage-1 model with the self/other weight
sharing of the superposition module removed.

SuperpositionNetworkMotionPleasureHoldNS = SuperpositionNetworkMotionPleasureHold, except that the
superposition module runs process-1 with `lstm` and process-2 with its own `lstm_other`.
`lstm_other` is a deep copy of `lstm` made after the whole model is built, so it starts from the same
initial weights and no random numbers are drawn: with the same seed the initial weights, batch order,
motion-hiding schedule and dropout masks are those of the shared (v13) run. The state layout
(state['self'], state['other']) and the parameter name prefix 'superposition_module' are unchanged, so
runners, freezing and analysis code work as before. share_lns (frozen) and the feature-prediction
module (detached input) stay shared (Sec.19.1).
"""

import copy

import torch

from .model_v12 import SuperpositionNetworkMotionPleasureHold
from .modules import LSTMState, SuperpositionModule


class SplitSuperpositionModule(SuperpositionModule):
    def __init__(self, shared):
        super(SuperpositionModule, self).__init__()   # nn.Module init only; take the built LSTM over
        self.config = shared.config
        self.lstm = shared.lstm
        self.lstm_other = copy.deepcopy(shared.lstm)

    def forward(self, sv, sm, ov, om):
        sh, sc = self.state['self']
        oh, oc = self.state['other']
        next_sh, next_sc = self.lstm(torch.cat([sv, sm], 1), (sh, sc))
        next_oh, next_oc = self.lstm_other(torch.cat([ov, om], 1), (oh, oc))
        self.state['self'] = LSTMState(next_sh, next_sc)
        self.state['other'] = LSTMState(next_oh, next_oc)
        return next_sh, next_oh


class SuperpositionNetworkMotionPleasureHoldNS(SuperpositionNetworkMotionPleasureHold):
    def __init__(self, config):
        super().__init__(config)
        old = self.superposition_module
        self.superposition_module = SplitSuperpositionModule(old)
        self.rnn_modules = [self.superposition_module if m is old else m for m in self.rnn_modules]
