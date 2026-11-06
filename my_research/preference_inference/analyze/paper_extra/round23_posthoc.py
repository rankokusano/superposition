"""Paper round 23, post-hoc descriptive breakdown (docs/paper_extra_log.md Sec.47.3; written after the results were read;
not part of the fixed procedure, not used for the reading). CPU. -> Round23/analysis/posthoc_E_by_side.json"""
# Round 23 post-hoc descriptive breakdown (not part of the fixed procedure; not used for the reading):
# M (E) split by the side of the selected E range: left-facing ranges 0,1,6,7 (where the red / green boxes usually
# appear in the training data) vs right-facing 2-5 (blue / cyan side). Saved per-episode values only.
import os, sys, json, numpy as np
PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PI, 'analyze/paper_extra'))
import vdir_stageA as VA
R23 = os.path.join(PI, 'data/result/paper_extra/Round23')
rg = np.load(os.path.join(R23, 'check/ranges.npz'))
E = rg['E']
grp = {'left0167': np.isin(E, [0, 1, 6, 7]), 'right2to5': np.isin(E, [2, 3, 4, 5])}
out = {'n_episodes': {k: int(v.sum()) for k, v in grp.items()}}
# fraction of training-like frames where pleasure is nonzero per range: use round 21 noswap A-1 views (all frames)
import h5py
sys.path.insert(0, PI)
from v11core.pleasure import pleasure_np
g = h5py.File(os.path.join(PI, 'data/data/paper_r21_swap/data_noswap.h5'), 'r')['test']
P = pleasure_np(g['self_vision'][:, :20].reshape(-1, 16, 64, 3))
out['frac_nonzero_pleasure_per_range_A1_views'] = (np.abs(P) > 0).mean(0).round(4).tolist()
for c in ('p50', 'p80', 'none', 'sr50'):
    for r in ('feat', 'img'):
        pp = {k: [] for k in grp}
        for s in range(6):
            acc = []
            for e in (160, 170, 180, 190, 200):
                z = np.load(os.path.join(R23, 'eval_dir', f'{c}_s{s}_ep{e}.npz'))
                acc.append(z[f'E_E_{r}'].astype(float) - z[f'zero_E_{r}'].astype(float))
            with np.errstate(all='ignore'):
                m = np.nanmean(np.stack(acc), 0)
                for k, msk in grp.items():
                    pp[k].append(float(np.nanmean(m[msk])))
        for k in grp:
            out[f'{c}|M|E|{r}|{k}'] = {'mean': float(np.mean(pp[k])), 't': VA.tsum(pp[k]), 'per_pair': pp[k]}
json.dump(out, open(os.path.join(R23, 'analysis/posthoc_E_by_side.json'), 'w'), indent=1)
for k, v in out.items():
    if isinstance(v, dict) and 'mean' in v:
        print(k, round(v['mean'], 4), [round(x, 4) for x in v['t']['ci95']])
    else:
        print(k, v)
