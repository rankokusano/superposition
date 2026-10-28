"""
v15 Sec.39 H4 (fewer observations): phase-2 data with only the first 100 train episodes of v15_s2_c2_p80_half_1000t_hideA2
(every key, episodes 0-99 in stored order) and the same test split (300 episodes, copied unchanged). Attributes copied,
plus 'derived_from' / 'n_train'. Checks that the copied arrays equal the source. CPU only.

    python analyze/v15/make_obs100_v15.py   # -> data/data/v15_s2_c2_p80_half_100t_hideA2/data.h5
"""
import os
import sys

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(_PI_DIR, 'data', 'data')
SRC, DST, N = 'v15_s2_c2_p80_half_1000t_hideA2', 'v15_s2_c2_p80_half_100t_hideA2', 100


def main():
    os.makedirs(os.path.join(DATA, DST), exist_ok=True)
    out = os.path.join(DATA, DST, 'data.h5')
    assert not os.path.exists(out), out
    with h5py.File(os.path.join(DATA, SRC, 'data.h5'), 'r') as f, h5py.File(out, 'w') as g:
        for k, v in f.attrs.items():
            g.attrs[k] = v
        g.attrs['derived_from'] = SRC
        g.attrs['n_train'] = N
        for split, n in (('train', N), ('test', None)):
            for k in f[split]:
                g.create_dataset(f'{split}/{k}', data=f[split][k][:n] if n else f[split][k][()])
    with h5py.File(os.path.join(DATA, SRC, 'data.h5'), 'r') as f, h5py.File(out, 'r') as g:
        ok = all(np.array_equal(g['train'][k][()], f['train'][k][:N]) for k in f['train']) and \
            all(np.array_equal(g['test'][k][()], f['test'][k][()]) for k in f['test'])
        print({s: {k: g[s][k].shape[0] for k in ('self_vision', 'layout')} for s in g}, 'equal to source:', ok, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
