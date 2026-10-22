"""
v12 thesis figure: the stage-2 attempts side by side (docs/v12_summary.md Sec.3). For each attempt (formal, 2b,
shape A, 2c on the hidden test) and pair: ① T2 and ② T5 at the late-5 checkpoints, the experienced value (mixed) and
the paired difference (mixed - random) with its CI, from the registered aggregations; D6 as a reference line.
CPU only. Writes data/result/v12/report/figs/s2_attempts.png.
"""

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.path.join(_PI, 'data', 'result', 'v12')
ATT = [('formal', 'stage2', 'aggregate_primary_s0s1s2s3.json'), ('2b', 'stage2b', 'aggregate_primary_s0s1s2s3.json'),
       ('shape A', 'stage2a', 'aggregate_primary_interim_s0.json'), ('2c (hidden test)', 'stage2c', None)]
D6 = {'green:sT2_t50': 0.60, 'pref:dT5': 0.42}


def load(out, fname):
    if fname:
        return json.load(open(os.path.join(V, out, 'eval', fname)))
    res = {}
    for f in sorted(glob.glob(os.path.join(V, out, 'eval', 'aggregate_primary_interim_s*.json'))):
        j = json.load(open(f))
        for k, v in j.items():
            if ':' in k:
                res.setdefault(k, {}).update({kk: vv for kk, vv in v.items() if kk.startswith('s')})
    return res


def main():
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    xt, xl = [], []
    x = 0
    for name, out, fname in ATT:
        agg = load(out, fname)
        seeds = sorted({k for v in agg.values() if isinstance(v, dict) for k in v if k[:1] == 's' and k[1:].isdigit()})
        for s in seeds:
            for ax, key in zip(axes, ('green:sT2_t50', 'pref:dT5')):
                if s not in agg.get(key, {}):
                    continue
                r = agg[key][s]
                m = [c['mixed'] for c in r['per_ckpt']]
                d = [c['diff'] for c in r['per_ckpt']]
                lo = [c['ci'][0] for c in r['per_ckpt']]
                hi = [c['ci'][1] for c in r['per_ckpt']]
                xs = x + np.linspace(-0.3, 0.3, len(m))
                ax.errorbar(xs, d, yerr=[np.array(d) - lo, np.array(hi) - d], fmt='o', color='tab:red', ms=3, lw=0.8,
                            label='paired diff (mixed - random), 95% CI' if x == 0 else None)
                ax.plot(xs, m, 'k_', ms=8, mew=2, label='experienced (mixed)' if x == 0 else None)
                ax.text(x, 0.27, f'{r["n_pass"]}/5', ha='center', fontsize=8)
            xt.append(x); xl.append(f'{name}\n{s}')
            x += 1
        x += 0.6
    for ax, key, ttl in zip(axes, ('green:sT2_t50', 'pref:dT5'), ('① T2 (swap moment)', '② T5 (only the preference differs)')):
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylim(-0.32, 0.31)
        ax.set_ylabel(ttl)
        ax.text(0.995, 0.02, f'D6 = {D6[key]:+.2f} (off scale)', transform=ax.transAxes, ha='right', fontsize=8)
        ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.0, 0.9))
    axes[1].set_xticks(xt); axes[1].set_xticklabels(xl, fontsize=7)
    fig.suptitle('v12 stage 2: the three attempts (late-5 checkpoints ep160-200; n/5 = checkpoints in the passing form)', fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(V, 'report', 'figs'), exist_ok=True)
    fig.savefig(os.path.join(V, 'report', 'figs', 's2_attempts.png'), dpi=120)
    print('saved')


if __name__ == '__main__':
    main()
