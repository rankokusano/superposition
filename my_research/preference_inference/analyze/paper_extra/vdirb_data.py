"""
Paper round 13 rev2 (V-dir stage B, docs/paper_extra_log.md Sec.37.5.2; pre-registered 4216497): description of the
new set paper_vdir_b_pref and the pre-registered data checks, from the data only (no model output):
cell counts, correlations, no-direction count, disagreeing episodes (per A-2 initial-y third), VIF of the main
regression; check (i) VIF(s_dir), VIF(s_pref) <= 5, check (ii) >= 100 disagreeing episodes. CPU only.
-> data/result/paper_extra/Vdir_B/data/summary.json
"""

import json
import os

import h5py
import numpy as np

import vdir_stageA as V
from vdir_stageB_check2 import wvif

PATH = os.path.join(V._PI, 'data', 'data', 'paper_vdir_b_pref', 'data.h5')
OUT = os.path.join(V.RES, 'paper_extra', 'Vdir_B', 'data')
TAU, EDGE = 1.0, 10.0 / 3.0
THIRDS = (('B', -np.inf, -EDGE), ('mid', -EDGE, EDGE), ('A', EDGE, np.inf))


def main():
    os.makedirs(OUT, exist_ok=True)
    f = h5py.File(PATH, 'r')
    g = f['test']
    lay, liked, lt = g['layout'][()], g['a2_liked'][()], g['layout_t'][()]
    op, sp = g['other_position'][()].astype(np.float64), g['self_position'][()].astype(np.float64)
    R = {'attrs': {k: (v.item() if hasattr(v, 'item') else str(v)) for k, v in f.attrs.items()}, 'n': int(len(lay)),
         'layout_t_constant': bool((lt == lay[:, None]).all())}
    red_dir = {L: float(np.sign(V.LANDMARKS[V.CODE_LAYOUT[L]]['Red'] @ V.NHAT)) for L in (0, 1)}
    s_red = np.array([red_dir[int(L)] for L in lay])
    s_pref = np.where(liked == 1, s_red, -s_red)
    y2, y1, x2, x1 = op[:, 0, 1], sp[:, 0, 1], op[:, 0, 0], sp[:, 0, 0]
    R['initial_range'] = {'a2': [op[:, 0].min(0).tolist(), op[:, 0].max(0).tolist()],
                          'a1': [sp[:, 0].min(0).tolist(), sp[:, 0].max(0).tolist()]}
    R['cells'] = {f"{'red' if lk else 'green'}|L{L}": int(((liked == lk) & (lay == L)).sum()) for lk in (1, 0) for L in (0, 1)}
    R['windows'] = {}
    for w, (_, (a, b)) in V.WINS.items():
        p = (op[:, b] - op[:, a - 1]) @ V.NHAT
        sd = np.sign(p)
        R['windows'][w] = {'n_no_dir': {str(t): int(((np.abs(p) < t) | (p == 0)).sum()) for t in V.TAUS}}
    p = (op[:, 50] - op[:, 0]) @ V.NHAT
    sd = np.sign(p)
    keep = (np.abs(p) >= TAU) & (p != 0)
    dis = keep & (sd != s_pref)
    R['main'] = {'n_dir': int(keep.sum()), 'n_no_dir': int((~keep).sum()), 'n_disagree': int(dis.sum()),
                 'pref_dir_agree': float((sd[keep] == s_pref[keep]).mean()),
                 'red_dir_agree': float((sd[keep] == s_red[keep]).mean())}
    R['by_third'] = {}
    for sn, lo, hi in THIRDS:
        t = (y2 >= lo) & (y2 < hi)
        R['by_third'][sn] = {'n': int(t.sum()), 'n_dir': int((t & keep).sum()), 'n_disagree': int((t & dis).sum())}
    R['cells_dir'] = {}
    for lk in (1, 0):
        for L in (0, 1):
            for sn, lo, hi in THIRDS:
                sel = (liked == lk) & (lay == L) & (y2 >= lo) & (y2 < hi)
                R['cells_dir'][f"{'red' if lk else 'green'}|L{L}|{sn}"] = {
                    'n': int(sel.sum()), 'A': int((sel & keep & (sd > 0)).sum()), 'B': int((sel & keep & (sd < 0)).sum()),
                    'none': int((sel & ~keep).sum())}
    Vs = {'s_pref': s_pref, 's_red': s_red, 's_dir': sd, 'p': p, 'a2y0': y2, 'a1y0': y1, 'a2x0': x2, 'a1x0': x1,
          'mag': np.linalg.norm(op[:, 50] - op[:, 0], axis=1)}
    ks = list(Vs)
    R['corr'] = {'vars': ks, 'r': [[float(np.corrcoef(Vs[a], Vs[b])[0, 1]) for b in ks] for a in ks]}
    Z = np.column_stack([s_pref, sd, s_red, y2, y1])[keep]
    R['vif_main'] = dict(zip(['pref', 'dir', 'red', 'a2y0', 'a1y0'], wvif(Z, np.ones(keep.sum()))))
    R['check_i_vif_le_5'] = bool(R['vif_main']['dir'] <= 5 and R['vif_main']['pref'] <= 5)
    R['check_ii_disagree_ge_100'] = bool(R['main']['n_disagree'] >= 100)
    json.dump(R, open(os.path.join(OUT, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: R[k] for k in ('n', 'layout_t_constant', 'cells', 'main', 'by_third', 'vif_main',
                                         'check_i_vif_le_5', 'check_ii_disagree_ge_100')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
