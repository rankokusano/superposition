"""
v15 stage 2 implementation check (docs/v15_experiment_log.md Sec.16), CPU, no training:
  1. model.compute_pleasure (torch, HoldV15) on stored views == the stored a1_pleasure (numpy, written by the
     collector from the same uint8 views) and a2_pleasure with A-2's values, on 5 episodes x 101 frames;
  2. both V15 classes build from their configs and run forward on 2 episodes (shapes, q2_hat for HoldOther).
"""
import os
import sys

import h5py
import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
import util  # noqa
from model import model_v15  # noqa
from v15core.pleasure import pleasure_torch, rank_values  # noqa
from v15core.world import COLORS  # noqa


def cfg(name):
    return util.load_config(os.path.join(_PI_DIR, 'config', 'model', name, 'default.yml'))


def main():
    g = h5py.File(os.path.join(_PI_DIR, 'data', 'data', 'v15_s2_c2_p80', 'data.h5'), 'r')['train']
    m = model_v15.SuperpositionNetworkMotionPleasureHoldV15(cfg('SuperpositionNetworkMotionPleasureHoldV15')).eval()
    worst1 = worst2 = 0.
    for n in range(5):
        sv = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][n], dim=4)))
        ov = torch.from_numpy(util.scale_vision(util.transpose_vision(g['other_vision'][n], dim=4)))
        p1 = m.compute_pleasure(sv).numpy()
        worst1 = max(worst1, np.abs(p1 - g['a1_pleasure'][n]).max())
        vals2 = rank_values(tuple(COLORS[c] for c in g['a2_rank'][n]))
        p2 = pleasure_torch((ov + 1) / 2, vals2).numpy()
        worst2 = max(worst2, np.abs(p2 - g['a2_pleasure'][n]).max())
    print('max |torch - stored| A-1 pleasure:', worst1, ' A-2 pleasure (A-2 values):', worst2)
    for name in ('SuperpositionNetworkMotionPleasureHoldV15', 'SuperpositionNetworkMotionPleasureHoldOtherV15'):
        mm = getattr(model_v15, name)(cfg(name)).eval()
        x = {'self_vision': torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][:2, 0], dim=4))),
             'self_motion': torch.from_numpy(g['self_motion'][:2, 0])}
        mm.init_state(2)
        with torch.no_grad():
            out = mm(x, p_mask_vision_self=0, p_mask_vision_other=0)
        print(name, {k: tuple(v.shape) for k, v in out.items()}, 'values', mm.values)


if __name__ == '__main__':
    main()
