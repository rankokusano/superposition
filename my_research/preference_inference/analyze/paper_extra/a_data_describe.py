"""
Paper extra A, data description and pairing checks (docs/paper_extra_log.md Sec.1.1, 2.1; data only, CPU).
For the p = 0.5 originals (v11_s2t_green, v11_s2_pref test) and the p = 0.8 sets (paper_*_a2p80):
  * A-2 within 5 of its liked colour's reference position (sim_v10.layout.LANDMARKS, as gate 6.1): all frames,
    late frames t = 50..100;
  * awake-step shares of a2_source (1 rule step, 2 rule turn -> random, 0 random turn) and the sleep share;
  * corner staying: late mean distance to the liked reference, late share within 2;
  * pairing with the original over all episodes: layout, a2_liked, A-1 position / motion / a1_source, A-2 sleep
    frames, initial A-2 position; the rule-turn subset property; A-1 / A-2 overlap frames (distance < 1).
-> data/result/paper_extra/A_cross/data_describe.json
"""

import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI)
from sim_v10 import layout as L  # noqa

DATA = os.path.join(_PI, 'data', 'data')
OUT = os.path.join(_PI, 'data', 'result', 'paper_extra', 'A_cross', 'data_describe.json')
PAIRS = {'green': ('v11_s2t_green', 'paper_s2t_green_a2p80'), 'pref': ('v11_s2_pref', 'paper_s2_pref_a2p80')}


def load(name):
    g = h5py.File(os.path.join(DATA, name, 'data.h5'), 'r')['test']
    return {k: g[k][()] for k in g if k not in ('self_vision', 'other_vision')}


def describe(d):
    liked = d['a2_liked'] if 'a2_liked' in d else np.zeros(len(d['layout']), np.uint8)
    ref = np.stack([L.LANDMARKS[L.CODE_LAYOUT[int(l)]]['Red' if lk == 1 else 'Green'] for l, lk in zip(d['layout'], liked)])
    dist = np.linalg.norm(d['other_position'] - ref[:, None, :], axis=-1)
    src = d['a2_source']
    awake = src != 3
    late = slice(50, 101)
    return {'near5_all': float((dist < 5).mean()), 'near5_late': float((dist[:, late] < 5).mean()),
            'late_mean_dist': float(dist[:, late].mean()), 'late_within2': float((dist[:, late] < 2).mean()),
            'awake_share_rule_step': float((src[awake] == 1).mean()), 'awake_share_rule_fallback': float((src[awake] == 2).mean()),
            'awake_share_random_turn': float((src[awake] == 0).mean()), 'sleep_share': float((~awake).mean()),
            'a1a2_overlap_frames_lt1': int((np.linalg.norm(d['self_position'] - d['other_position'], axis=-1) < 1).sum())}


def main():
    res = {}
    for k, (o_name, n_name) in PAIRS.items():
        o, n = load(o_name), load(n_name)
        same = {q: bool(np.array_equal(o[q], n[q])) for q in ('layout', 'self_position', 'self_motion', 'a1_source', 'layout_t')}
        if 'a2_liked' in o:
            same['a2_liked'] = bool(np.array_equal(o['a2_liked'], n['a2_liked']))
        same['a2_sleep_frames'] = bool(np.array_equal(o['a2_source'] == 3, n['a2_source'] == 3))
        same['a2_initial_position'] = bool(np.array_equal(o['other_position'][:, 0], n['other_position'][:, 0]))
        turn5, turn8 = np.isin(o['a2_source'], [1, 2]), np.isin(n['a2_source'], [1, 2])
        same['rule_turns_05_subset_of_08'] = bool(np.all(~turn5 | turn8))
        same['n_episodes_a2_path_differs'] = int((~np.all(o['other_position'] == n['other_position'], axis=(1, 2))).sum())
        res[k] = {'p05': describe(o), 'p08': describe(n), 'pairing': same}
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
