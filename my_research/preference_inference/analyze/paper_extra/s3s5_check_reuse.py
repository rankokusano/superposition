"""
Paper round 6 (docs/paper_extra_log.md Sec.34; instructions Sec.1.5): pairs 1-3 values that the 6-pair analysis
reuses, recomputed from the per-episode result files with c_aggregate.py's / b4_aggregate.py's own functions and
compared with the saved summaries (C_signrand/summary.json, B4/summary.json). Prints the late-5 means and the ranges
that the paper quotes (results.tex Sec.4.3, supplement S4), for checking by eye against the text. CPU, no model run.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import c_aggregate as CA  # noqa
import b4_aggregate as BA  # noqa

EPS = CA.EPS
S = json.load(open(os.path.join(CA.RES, 'paper_extra', 'C_signrand', 'summary.json')))['signrand']
B = json.load(open(os.path.join(CA.RES, 'paper_extra', 'B4', 'summary.json')))


def late5(c, s):
    return {'G5b': [float(np.nanmean(CA.g5(c, s, e, 'G5b'))) for e in EPS],
            'signfollow_green': [float(np.nanmean(CA.sf(c, s, e, 'green'))) for e in EPS],
            'signfollow_pref': [float(np.nanmean(CA.sf(c, s, e, 'pref'))) for e in EPS],
            'G3_hold': [float(np.nanmean(CA.g3(c, s, e, 'hold'))) for e in EPS],
            'G3_dP': [float(np.nanmean(CA.g3(c, s, e, 'dP'))) for e in EPS],
            'G1_visible': [CA.trial_json(c, s, e)['tracking']['p_next']['visible'] for e in EPS],
            'BA_L1_l1late5': [CA.inv(c, s, e, 'L1') for e in EPS],
            'BA_FP_mselate5': [CA.inv(c, s, e, 'FP') for e in EPS]}


def main():
    ok, rows = True, {}
    for s in range(3):
        for c in ('signrand', 'mixed80'):
            v, ref = late5(c, s), S['per_pair'][f's{s}']['values'][c]
            for q, x in v.items():
                same = x == ref[q]
                ok &= same
                rows.setdefault(q, {}).setdefault(c, []).append((float(np.mean(x)), min(x), max(x), same))
    print('C_signrand/summary.json reproduced exactly:', ok)
    for q, d in rows.items():
        for c, r in d.items():
            print(f'{q:16s} {c:9s} late-5 means', ' / '.join(f'{m:+.3f}' for m, *_ in r),
                  '| point range', f'{min(a for _, a, _, _ in r):.3f}..{max(b for _, _, b, _ in r):.3f}', '| same', [t for *_, t in r])
    print('C judgement counts', S['judge_counts'])
    qs = {'G5b': lambda c, s, e: BA.g5b(c, s, e), 'signfollow_green': lambda c, s, e: BA.sf(c, s, e, 'green'),
          'signfollow_pref': lambda c, s, e: BA.sf(c, s, e, 'pref')}
    okb = True
    for s in range(3):
        P = B['per_pair'][f's{s}']
        for c in ('signrand', 'mixed80', 'random'):
            for q, fn in qs.items():
                okb &= [float(np.nanmean(fn(c, s, e))) for e in EPS] == P['values'][c][q]
            tr = [json.load(open(os.path.join(BA.B4, 'track', f'{c}_s{s}_ep{e}.json')))['p_next']['visible'] for e in EPS]
            okb &= tr == P['values'][c]['track_visible']
        for q, fn in qs.items():
            r = BA.pointwise(lambda c, e: fn(c, s, e), 'signrand', 'mixed80', -1)
            okb &= r['n_ok'] == P[f'{q}_sr_vs80']['n_ok']
            diff = np.mean(P['values']['signrand'][q]) - np.mean(P['values']['mixed80'][q])
            print(f'B4 s{s} {q:16s} sr-80 points {r["n_ok"]}/5  mean-of-means diff {diff:+.3f}')
        print(f'B4 s{s} track_visible 80% {min(P["values"]["mixed80"]["track_visible"]):.3f}..{max(P["values"]["mixed80"]["track_visible"]):.3f}')
    print('B4/summary.json reproduced exactly:', okb)


if __name__ == '__main__':
    main()
