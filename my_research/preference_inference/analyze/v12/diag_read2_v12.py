"""v12 check B (docs/v12_experiment_log.md Sec.20): compare A-2 read-outs on the predicted images stored by
diag_read_v12.py (P-c ep150, G5a setting, set 1, 40 episodes x 15 steps): direction error to the true A-2 and
fraction readable, for (a) the registered absolute-darkness read-out (tau 0.2, mass >= 2) and (b) a
background-relative read-out (darkness below the row median of the same predicted image)."""
import sys
sys.path.insert(0, 'analyze/v11'); sys.path.insert(0, '.')
import numpy as np, h5py
from eval_s2_v11 import load_set
from v11core.pleasure import COL_ANGLES

z = np.load('data/result/v12/diag_read/diag_pred.npz')
d = load_set('green')
sp = d['self_position'][:40, 21:36]; op = d['other_position'][:40, 21:36]
true_ang = np.arctan2(op[..., 1] - sp[..., 1], op[..., 0] - sp[..., 0])


def angle(w):
    col = w.sum(-2)                       # (..., W)
    M = col.sum(-1)
    z_ = (col * np.exp(1j * COL_ANGLES)).sum(-1)
    return np.angle(z_), M


def err(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


for c in ('random', 'mixed', 'truth'):
    P = z[c] if c != 'truth' else z['truth'].astype(np.float32) / 255
    mx = P.max(-1)
    for nm, w in (('abs tau0.2', np.clip((0.2 - mx) / 0.2, 0, 1)),
                  ('rowmedian-0.05', np.clip(np.median(mx, axis=-1, keepdims=True) - mx - 0.05, 0, None)),
                  ('rowmedian-0.10', np.clip(np.median(mx, axis=-1, keepdims=True) - mx - 0.10, 0, None))):
        a, M = angle(w)
        thr = 2 if nm.startswith('abs') else 0.5
        ok = M >= thr
        e = np.degrees(err(a, true_ang))
        print(f'{c:6s} {nm:15s} readable {ok.mean():.2f}  direction error median {np.median(e[ok]):6.1f} deg, '
              f'frac < 30 deg {np.mean(e[ok] < 30):.2f}')
