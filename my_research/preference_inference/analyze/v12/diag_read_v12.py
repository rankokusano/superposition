"""v12 check B (docs/v12_experiment_log.md Sec.20): why the A-2 position cannot be read from many predicted images (G5a setting, P-c ep150, set 1, first 40 episodes, CPU; diagnostic only)."""
import sys, os
sys.path.insert(0, 'analyze/v12'); sys.path.insert(0, 'analyze/v11'); sys.path.insert(0, '.')
import numpy as np, torch, h5py
from eval_trial_v12 import load_v12
from eval_s2_v11 import load_set, true_pleasure
from p2test_v11 import calibrate, read_a2
import util
torch.set_num_threads(8)
dev = torch.device('cpu')
d = load_set('green'); PA2 = true_pleasure('green', d)
g = h5py.File(d['path'], 'r')['test']
NE = 40
sv = g['self_vision'][:NE]
alpha, beta = calibrate(g['self_vision'][:, :100:5], d['self_position'][:, :100:5], d['other_position'][:, :100:5])
out = {}
for c in ('random', 'mixed'):
    model, _ = load_v12(f'v12_trial_pc_{c}_mse', 0, 150, dev); model.motion_masking = False
    x = torch.from_numpy(util.scale_vision(util.transpose_vision(sv))); m = torch.from_numpy(g['self_motion'][:NE]).float()
    util.seed_all(0); model.init_state(NE)
    P = []
    with torch.no_grad():
        for t in range(35):
            q = torch.zeros(NE, 11); q[:, 10] = 1; q[:, 2:10] = torch.from_numpy(PA2[:NE, min(t, 20)])
            pr = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0 if t == 0 else 1, 0 if t == 0 else 1, p2_override=q)['self_vision']
            if t >= 20: P.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy())
    P = np.stack(P, 1)   # (NE, 15, 16, 64, 3)
    mx = P.max(-1)
    tv = sv[:, 21:36].astype(np.float32) / 255
    out[c] = P
    print(c, 'pred: min of channel-max per image  pct 5/25/50/75:', np.percentile(mx.min((2, 3)), [5, 25, 50, 75]).round(3))
    print(c, 'true next view: min channel-max pct:', np.percentile(tv.max(-1).min((2, 3)), [5, 25, 50]).round(3))
    for tau in (0.2, 0.3, 0.4, 0.5):
        w = np.clip((tau - mx) / tau, 0, 1); M = w.sum((2, 3))
        print(c, f'tau {tau}: frac M>=2 {np.mean(M >= 2):.2f}, frac M>=0.5 {np.mean(M >= 0.5):.2f}')
    # darkness at A-2's true column in the true view vs prediction
    tb = (tv.max(-1) < 0.1)                       # true black pixels (A-2)
    has = tb.sum((2, 3)) >= 3
    colmask = tb.any(2)                            # (NE, 15, 64) columns where A-2 truly is
    pm = mx.min(2)                                 # darkest value per column in prediction
    print(c, 'frames with A-2 visible in truth:', has.mean().round(2),
          '| pred darkest value in A-2 columns (median):', np.median(pm[colmask]).round(3), 'elsewhere:', np.median(pm[~colmask]).round(3))
np.savez_compressed('data/result/v12/diag_read/diag_pred.npz', **out, truth=sv[:, 21:36])
