"""
Thesis figure: v11 vs v12 on one page (docs/v12_summary.md Sec.9). From saved outputs only (CPU):
  (a) stage-1 M1(d) paired difference (mixed - random), late-5 mean per pair (v11 mse gate table / v12 G2 rows);
  (b) colour criterion (S_red - S_green on swapped episodes), experienced model, late-5 mean per pair;
  (c) own-side accumulation g (input-stop test) of the experienced model per pair: v11 localise (input zeroed,
      K = 20, ep200) / v12 G3 (pleasure held, K = 15, late-5 mean);
  (d) other-side accumulation G5a g_other of the experienced model (set 1, K = 15, freeze, late-5 mean); v11 had
      no G5a (its process-2 reference test did not give the registered positive form);
  (e) stage-2 judgements (pairs in the passing form / pairs evaluated) as a table.
Writes data/result/v12/report/figs/compare_v11_v12.png.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(_PI, 'data', 'result')
SEEDS = (0, 1, 2)
EPS = (160, 170, 180, 190, 200)
C11, C12 = '#7f7f7f', '#d62728'


def v11_stage1():
    t = json.load(open(os.path.join(R, 'v11', 'main', 'gate', 'gate_mse_late5.json')))['table']
    m1d = [t['c_cf_diff']['per_seed_late5_mean'][str(s)] for s in SEEDS]
    col = [t['colour_mixed']['per_seed_late5_mean'][str(s)] for s in SEEDS]
    loc = json.load(open(os.path.join(R, 'v11', 'main', 'localise', 'localise_ep200.json')))
    g = [loc[f'mixed_s{s}']['zero_K20']['g'] for s in SEEDS]
    return m1d, col, g


def v12_stage1():
    rows = json.load(open(os.path.join(R, 'v12', 'main', 's1eval', 'gate_s1.json')))['per_seed']
    def mean(key, s):  # noqa
        return float(np.mean([r['values'][key] for r in rows[str(s)]['rows']]))
    m1d = [mean('M1d_paired', s) for s in SEEDS]
    col = [mean('colour_mixed', s) for s in SEEDS]
    g3 = [mean('G3_mixed', s) for s in SEEDS]
    g5 = []
    for s in SEEDS:
        v = [json.load(open(os.path.join(R, 'v12', 'main', 'g5', f's{s}', f'g5_mixed_ep{e}.json')))['G5a']['green_K15_freeze']['g_other']
             for e in EPS]
        g5.append((np.mean([x['mean'] for x in v]), np.mean([x['ci95'][0] for x in v]), np.mean([x['ci95'][1] for x in v])))
    return m1d, col, g3, g5


def bars(ax, vals11, vals12, title, ylab, err11=None, err12=None):
    x = np.arange(len(SEEDS))
    if vals11 is not None:
        ax.bar(x - 0.18, vals11, 0.34, color=C11, label='v11', yerr=err11, capsize=3)
    ax.bar(x + 0.18, vals12, 0.34, color=C12, label='v12', yerr=err12, capsize=3)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f's{s}' for s in SEEDS])
    ax.set_title(title, fontsize=9); ax.set_ylabel(ylab, fontsize=8)
    ax.legend(fontsize=7, loc='lower right' if title.startswith('(c)') else ('upper right' if title.startswith('(d)') else 'upper left'))


def main():
    m1d11, col11, g11 = v11_stage1()
    m1d12, col12, g312, g512 = v12_stage1()
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.75])
    ax = fig.add_subplot(gs[0, 0])
    bars(ax, m1d11, m1d12, '(a) value-action: M1(d) paired diff\n(mixed - random, late-5;\nv12 on motion-hidden steps)', 'pull toward "one step to red"')
    ax = fig.add_subplot(gs[0, 1])
    bars(ax, col11, col12, '(b) value-colour: S_red - S_green\n(experienced, swapped episodes,\nlate-5)', 'S_red - S_green')
    ax = fig.add_subplot(gs[0, 2])
    e11 = np.array([[x['mean'] - x['ci95'][0] for x in g11], [x['ci95'][1] - x['mean'] for x in g11]])
    bars(ax, [x['mean'] for x in g11], g312, '(c) own-side accumulation g (input stop)\nv11 FAIL 0/3 (input zeroed, K=20, ep200)\nv12 PASS 3/3 (G3: pleasure held, K=15, late-5)', 'g (red slope - others)',
         err11=e11)
    ax = fig.add_subplot(gs[0, 3])
    e12 = np.array([[m - lo for m, lo, hi in g512], [hi - m for m, lo, hi in g512]])
    bars(ax, None, [m for m, lo, hi in g512], '(d) other-side accumulation G5a g_other\nv12 PASS 3/3 (experienced, set 1, K=15,\nfreeze, late-5, 95% CI)', 'g_other (liked - others)', err12=e12)
    ax.set_ylim(0, 0.36); ax.text(0.30, 0.97, 'v11: not observed\n(reference test only)', transform=ax.transAxes, va='top', fontsize=8)
    ax = fig.add_subplot(gs[1, :]); ax.axis('off')
    rows = [['', 'v11', 'v12 formal', 'v12 2b (shaped output)', 'v12 2c (A-2 hidden; hidden test)'],
            ['stage 2 ① T2 (swap moment)', 'FAIL 1/3', 'FAIL 1/4', 'FAIL 0/4', 'FAIL 0/3 (reversed)'],
            ['stage 2 ② T5 (only the preference differs)', 'PASS 2/3 (1/10 of D6)', 'FAIL 1/4', 'FAIL 1/4', 'FAIL 1/3'],
            ['what G sent', 'bins 2-5 (79-94% of var.)', 'bins 2-5 (71-86%)', 'u fixed (3/4 pairs), strengths per bin', 'u mostly fixed, strengths per bin'],
            ['self/other code alignment', '0.16-0.30 (Fig.5 fails)', 'v4 encoders frozen (given): 0.80', '(same)', '(same)']]
    tb = ax.table(cellText=rows, loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.6)
    for j in range(5):
        tb[0, j].set_facecolor('#eeeeee')
    ax.set_title('(e) stage 2 (other): pairs in the passing form / pairs evaluated', fontsize=9)
    fig.suptitle('v11 vs v12: stage 1 (self, experienced model) and stage 2 (other)', fontsize=11)
    fig.tight_layout()
    out = os.path.join(R, 'v12', 'report', 'figs', 'compare_v11_v12.png')
    fig.savefig(out, dpi=120)
    print('saved', out)


if __name__ == '__main__':
    main()
