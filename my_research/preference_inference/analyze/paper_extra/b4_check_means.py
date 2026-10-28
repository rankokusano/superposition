"""
Paper extra B4: why the late-5 mean of the paired difference differs from the difference of the model means
(docs/paper_extra_log.md Sec.29.10; answer to Kusano-san's question, no new computation). For each pair, point and
contrast: n readable episodes per model, n in both, the paired mean (eval_g5_v12.summ: NaN episodes dropped from the
difference), each model's own nanmean, and each model's mean restricted to the episodes readable in both.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import b4_aggregate as ba  # noqa

for s in range(3):
    for a, b in (('signrand', 'mixed80'), ('signrand', 'random'), ('mixed80', 'random')):
        rows = []
        for e in ba.EPS:
            x, y = ba.g5b(a, s, e), ba.g5b(b, s, e)
            both = np.isfinite(x) & np.isfinite(y)
            rows.append((np.isfinite(x).sum(), np.isfinite(y).sum(), both.sum(), ba.summ(x - y)['mean'],
                         np.nanmean(x) - np.nanmean(y), x[both].mean() - y[both].mean(), x[both].mean(), y[both].mean()))
        r = np.array(rows, float)
        print(f'pair {s + 1} G5b {a}-{b}: n {r[:, 0].min():.0f}-{r[:, 0].max():.0f} / {r[:, 1].min():.0f}-{r[:, 1].max():.0f}, '
              f'both {r[:, 2].min():.0f}-{r[:, 2].max():.0f} | late5 paired {r[:, 3].mean():+.3f}  diff of own means {r[:, 4].mean():+.3f}  '
              f'diff of means on common eps {r[:, 5].mean():+.3f} (={r[:, 6].mean():.3f} - {r[:, 7].mean():.3f})')


def gap(fx, fy, eps=ba.EPS):
    """late-5 mean of the paired difference, difference of the own nanmeans, max |gap| over points."""
    p, o = [], []
    for e in eps:
        x, y = fx(e), fy(e)
        p.append(ba.summ(x - y)['mean'])
        o.append(np.nanmean(x) - np.nanmean(y))
    return np.mean(p), np.mean(o)


print('--- all contrasts: late-5 paired mean vs difference of own means (B4 and the common test of Sec.25/28)')
sys.path.insert(0, os.path.join(_HERE))
import c_aggregate as ca  # noqa
for s in range(3):
    for q in ('G5b', 'green', 'pref'):
        fb = (lambda c, e: ba.g5b(c, s, e)) if q == 'G5b' else (lambda c, e, q=q: ba.sf(c, s, e, q))
        fc = (lambda c, e: ca.g5(c, s, e, 'G5b')) if q == 'G5b' else (lambda c, e, q=q: ca.sf(c, s, e, q))
        out = []
        for a, b in (('signrand', 'mixed80'), ('signrand', 'random'), ('mixed80', 'random')):
            pb, ob = gap(lambda e: fb(a, e), lambda e: fb(b, e))
            pc, oc = gap(lambda e: fc(a, e), lambda e: fc(b, e))
            out.append(f'{a[:2]}-{b[:2]} B4 {pb:+.3f}/{ob:+.3f} CT {pc:+.3f}/{oc:+.3f}')
        for a in ('mixed80', 'random'):
            pc, oc = gap(lambda e: fc('pzero', e), lambda e: fc(a, e))
            out.append(f'pz-{a[:2]} CT {pc:+.3f}/{oc:+.3f}')
        print(f'pair {s + 1} {q:5s}', ' | '.join(out))
