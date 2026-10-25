"""
v15 stage 2 (docs/v15_experiment_log.md Sec.16): the v12 stage-1 / stage-2 models with the v15 pleasure:
P[k] = sum_c value_c * (#pixels of colour c in camera bin k) / 128 (gray = 0; v15core.pleasure), with A-1's
fixed rank values from the model config `pleasure.values` (Red 1, Blue 0.5, Cyan 0, Green -1). Nothing else
changes (hold rule, hide schedule, G, process-2 form); `pleasure.liked/disliked` stay in the config only because
the v12 base class reads them, and are unused.
"""

from v15core.pleasure import pleasure_torch as pleasure_torch_values

from .model_v12 import SuperpositionNetworkMotionPleasureHold, SuperpositionNetworkMotionPleasureHoldOther


class _ValuePleasure(object):
    def _init_values(self, config):
        self.values = {k: float(v) for k, v in dict(config.pleasure['values']).items()}

    def compute_pleasure(self, sv):
        """sv: (B, 3, H, W) scaled to [-1, 1] by the data loader."""
        return self.pleasure_scale * pleasure_torch_values((sv + 1) / 2, self.values)


class SuperpositionNetworkMotionPleasureHoldV15(_ValuePleasure, SuperpositionNetworkMotionPleasureHold):
    def __init__(self, config):
        super().__init__(config)
        self._init_values(config)


class SuperpositionNetworkMotionPleasureHoldOtherV15(_ValuePleasure, SuperpositionNetworkMotionPleasureHoldOther):
    def __init__(self, config):
        super().__init__(config)
        self._init_values(config)
