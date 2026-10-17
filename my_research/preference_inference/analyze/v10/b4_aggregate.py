"""
v10 B4 aggregation (docs/v10_experiment_log.md Sec.5.6, 5.17, 5.22 (0), 5.26 (1), 5.29, 5.30).

Per model x seed: late-5 (ep360-400) mean +- inter-checkpoint sd of every
metric. Per model: (1) every seed individually, (2) mean +- sd over ALL seeds,
(3) mean +- sd over CONVERGED seeds (B3-frozen: seed1 only = "reference based
on a single run"). Main gates on B3-ft (converged = all 3), with B3-frozen
reported as the comparison condition; R3-A_400 = seeds 0, 1.

Bootstrap (1000, RandomState(0), percentile 95% CI) over test episodes: each
episode's correctness is averaged over the model's seeds x late-5 checkpoints;
differences are paired by episode.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, git_hash  # noqa

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

import argparse
B4 = os.path.join(OUT_ROOT, 'b4')
TAG = ''
EPOCHS = [360, 370, 380, 390, 400]
TIMES = [1, 5, 10, 20, 50, 100]
MODELS = {
    'B3-ft': dict(exp='v10_b3_ft', seeds=[0, 1, 2], converged=[0, 1, 2]),
    'B3-frozen': dict(exp='v10_b3_frozen', seeds=[0, 1, 2], converged=[1]),
    'R3-A_400': dict(exp='r3_a_direct_1000pretrain_400', seeds=[0, 1], converged=[0, 1]),
}
COL = {'B3-ft': '#E45756', 'B3-frozen': '#4C78A8', 'R3-A_400': '#72B7B2'}
N_BOOT = 1000


def flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f'{pre}{k}/'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f'{pre}{k}'] = float(v)
    return out


def ckpt_metrics(exp, s, e):
    d = os.path.join(B4, 'eval', exp, f's{s}', f'ep{e}')
    t = json.load(open(os.path.join(d, f'test_{TAG or "swapmix"}.json')))
    m = {}
    m.update(flat(t['readout_accuracy'], 'readout/'))
    m.update(flat(t['distance_r2'], 'distance_r2/'))
    m.update(flat(t['vision_error'], 'vision_error/'))
    if not TAG:
        for ds, tag in (('v10_swapmix_r2', 'r2_indist'), ('r2_a1random_a2rl', 'r2_common')):
            m.update(flat(json.load(open(os.path.join(d, f'r2_{ds}.json')))['r2'], f'{tag}/'))
    corr = np.load(os.path.join(d, f'test_{TAG or "swapmix"}_correct.npz'))
    return m, corr


def boot_ci(diff_per_ep, rng):
    n = len(diff_per_ep)
    bs = np.array([diff_per_ep[rng.randint(0, n, n)].mean() for _ in range(N_BOOT)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def ep_correct(store, model, seeds, key):
    return np.mean([store[model][s][e][key] for s in seeds for e in EPOCHS], axis=0)


def main():
    global B4, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='', help="'' = B4 on v10_swapmix_r2; else re-evaluation tag (e.g. a2v3)")
    TAG = ap.parse_args().tag
    if TAG:
        B4 = os.path.join(OUT_ROOT, 'b4_reeval', TAG)
    metrics, store, layout = {}, {}, None
    for name, M in MODELS.items():
        metrics[name], store[name] = {}, {}
        for s in M['seeds']:
            rows, store[name][s] = [], {}
            for e in EPOCHS:
                m, c = ckpt_metrics(M['exp'], s, e)
                rows.append(m)
                store[name][s][e] = {k: c[k].astype(float) for k in c.files if k not in ('layout', 'ep_vision_error')}
                layout = c['layout']
            metrics[name][s] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1)))
                                for k in rows[0]}
    keys = list(metrics['B3-ft'][0].keys())
    summary = {}
    for name, M in MODELS.items():
        summary[name] = {}
        for k in keys:
            per = {str(s): {'late5_mean': metrics[name][s][k][0], 'late5_sd': metrics[name][s][k][1]} for s in M['seeds']}
            allv = [metrics[name][s][k][0] for s in M['seeds']]
            cv = [metrics[name][s][k][0] for s in M['converged']]
            summary[name][k] = {'per_seed': per,
                                'all_seeds_mean': float(np.mean(allv)), 'all_seeds_sd': float(np.std(allv, ddof=1)) if len(allv) > 1 else None,
                                'converged_mean': float(np.mean(cv)), 'converged_sd': float(np.std(cv, ddof=1)) if len(cv) > 1 else None,
                                'converged_seeds': M['converged']}

    gates = {}
    rng = np.random.RandomState(0)
    r3a = ep_correct(store, 'R3-A_400', [0, 1], 'h1_t20')
    for name, seedsets in (('B3-ft', {'converged (= all 3)': [0, 1, 2]}),
                           ('B3-frozen', {'all seeds': [0, 1, 2], 'converged seed1 only (reference, single run)': [1]})):
        gates[name] = {}
        for lab, seeds in seedsets.items():
            h1 = ep_correct(store, name, seeds, 'h1_t20')
            pq = ep_correct(store, name, seeds, 'probe_q_t20')
            enc = ep_correct(store, name, seeds, 'sv_enc_t20')
            g1_diff = h1 - r3a
            g = {'seeds': seeds,
                 'acc_h1_t20': float(h1.mean()), 'acc_r3a_h1_t20': float(r3a.mean()),
                 'gate1_diff_vs_r3a': float(g1_diff.mean()), 'gate1_diff_ci95': boot_ci(g1_diff, rng),
                 'acc_probe_q_t20': float(pq.mean()), 'acc_sv_enc_t20': float(enc.mean()),
                 'gate2_diff_h1_minus_probe_q': float((h1 - pq).mean()), 'gate2_ci_h1_minus_probe_q': boot_ci(h1 - pq, rng),
                 'gate2_diff_h1_minus_sv_enc': float((h1 - enc).mean()), 'gate2_ci_h1_minus_sv_enc': boot_ci(h1 - enc, rng)}
            g['gate1_pass'] = bool(g['acc_h1_t20'] >= 0.80 and g['gate1_diff_vs_r3a'] >= 0.15 and g['gate1_diff_ci95'][0] > 0)
            g['gate2_pass'] = bool(g['gate2_ci_h1_minus_probe_q'][0] > 0 and g['gate2_ci_h1_minus_sv_enc'][0] > 0)
            g['reading'] = ('colour-organised internal state' if g['gate1_pass'] and g['gate2_pass'] else
                            'colour info in h1 but indistinguishable from passing probe-Q / Encoder output' if g['gate1_pass'] else
                            'not organised by colour')
            if TAG:  # Sec.5.34: compare with reading the layout from A-2's TRUE position (t=20)
                a2c = a2_position_correct(20)
                g['acc_A2_true_position_t20'] = float(a2c.mean())
                g['diff_h1_minus_A2_position'] = float((h1 - a2c).mean())
                g['ci_h1_minus_A2_position'] = boot_ci(h1 - a2c, rng)
                g['h1_beats_A2_position'] = bool(g['ci_h1_minus_A2_position'][0] > 0)
            gates[name][lab] = g
    if TAG:
        a2c = a2_position_correct(20)
        r3 = r3a
        gates['R3-A_400'] = {'seeds 0,1': {'acc_h1_t20': float(r3.mean()), 'acc_A2_true_position_t20': float(a2c.mean()),
                                           'diff_h1_minus_A2_position': float((r3 - a2c).mean()),
                                           'ci_h1_minus_A2_position': boot_ci(r3 - a2c, rng)}}
    out = {'git_commit': git_hash(), 'epochs': EPOCHS, 'n_boot': N_BOOT, 'n_test_episodes': int(len(layout)),
           'n_original': int((layout == 0).sum()), 'n_swap': int((layout == 1).sum()),
           'gates': gates, 'summary': summary}
    out['dataset'] = f'v10_swapmix_eval_{TAG}' if TAG else 'v10_swapmix_r2'
    json.dump(out, open(os.path.join(B4, 'b4_summary.json'), 'w'), indent=2)
    print(json.dumps(gates, indent=1))
    plots(summary)


def a2_position_correct(t):
    import h5py
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from layout_readout_v10 import folds
    from common_v10 import PI_DIR
    f = h5py.File(os.path.join(PI_DIR, 'data', 'data', f'v10_swapmix_eval_{TAG}', 'data.h5'), 'r')['test']
    lay, X = f['layout'][()].astype(int), f['other_position'][:, t - 1]
    pred = np.zeros(len(lay))
    for tr, te in folds(np.arange(len(lay))):
        pred[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=10000)).fit(X[tr], lay[tr]).predict(X[te])
    return (pred == lay).astype(float)


def mean_sd(summary, name, k, conv=False):
    v = summary[name][k]
    return (v['converged_mean'], v['converged_sd']) if conv else (v['all_seeds_mean'], v['all_seeds_sd'])


def plots(summary):
    # 1. layout read-out over time
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True, sharey=True)
    ax = axes[0]
    for name in MODELS:
        m = [summary[name][f'readout/h1/{t}']['all_seeds_mean'] for t in TIMES]
        ax.plot(TIMES, m, 'o-', color=COL[name], lw=2, label=f'{name} h1 (all seeds)')
        for s, v in summary[name][f'readout/h1/20']['per_seed'].items():
            ax.scatter([20], [v['late5_mean']], s=14, color=COL[name], edgecolors='k', zorder=3)
    ax.set_title('layout read-out from h1', fontsize=10)
    for ax, name in zip(axes[1:], ('B3-ft', 'B3-frozen')):
        for k, ls, lab in (('h1', '-', 'h1'), ('probe_q', '--', 'probe-Q only'), ('sv_enc', ':', 'Encoder output only')):
            m = [summary[name][f'readout/{k}/{t}']['all_seeds_mean'] for t in TIMES]
            ax.plot(TIMES, m, 'o' + ls, color=COL[name], lw=2, label=lab)
        ax.set_title(f'{name}: h1 vs its inputs (all seeds)', fontsize=10)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(TIMES)
        ax.set_xticklabels(TIMES)
        ax.axhline(0.5, color='0.6', lw=0.8)
        ax.axhline(0.8, color='0.6', lw=0.8, ls='--')
        ax.axvline(20, color='0.85', lw=6, zorder=0)
        ax.set_xlabel('t (input frames processed)')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(color='0.93')
    axes[0].set_ylabel('accuracy (test split, 5-fold by episode, late-5)')
    fig.suptitle(f'v10 B4{" re-evaluation " + TAG if TAG else ""}: reading the layout from the shared module; shaded = t=20 (main gates)')
    fig.savefig(os.path.join(B4, 'b4_layout_readout.png'), dpi=110)
    plt.close(fig)

    # 2. distance R^2 (red vs top-left) per subset
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True, sharey=True)
    for ax, sub in zip(axes, ('all', 'swap_rg', 'original') if TAG else ('swap_rg', 'original', 'all')):
        x = np.arange(len(MODELS))
        for i, (k, hatch) in enumerate((('dist_red', ''), ('dist_top_left', '//'))):
            m = [summary[n][f'distance_r2/{sub}/{k}']['all_seeds_mean'] for n in MODELS]
            ax.bar(x + (i - 0.5) * 0.38, m, 0.35, color=[COL[n] for n in MODELS], hatch=hatch, edgecolor='k',
                   label='distance to red' if i == 0 else 'distance to top-left (-9,9)')
            for j, n in enumerate(MODELS):
                pts = [v['late5_mean'] for v in summary[n][f'distance_r2/{sub}/{k}']['per_seed'].values()]
                ax.scatter([x[j] + (i - 0.5) * 0.38] * len(pts), pts, s=12, c='k', zorder=3)
                ax.text(x[j] + (i - 0.5) * 0.38, m[j] + 0.02, f'{m[j]:.2f}', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(list(MODELS))
        ax.set_title({'swap_rg': 'swapped episodes' + (' (reference)' if TAG else ' (main contrast)'),
                      'original': 'original episodes (identical targets)',
                      'all': 'all test episodes' + (' (main contrast, reference only; Sec.5.32)' if TAG else '')}[sub], fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('R^2 from h1 (all seeds mean; dots = seeds)')
    fig.suptitle('v10 B4: does h1 encode distance to red or distance to the top-left place?')
    fig.savefig(os.path.join(B4, 'b4_distance_r2.png'), dpi=110)
    plt.close(fig)

    # 3. 4-axis R^2 (common / in-dist all / in-dist per layout) and 4. vision error
    if TAG:
        return vision_plot(summary)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), constrained_layout=True, sharey=True)
    ax_names = ['h1_to_self', 'h1_to_other', 'h2_to_self', 'h2_to_other']
    for ax, (pre, ttl) in zip(axes, (('r2_common/all', 'common: r2_a1random_a2rl'), ('r2_indist/all', 'in-dist: v10_swapmix_r2'),
                                     ('r2_indist/original', 'in-dist, original episodes'), ('r2_indist/swap_rg', 'in-dist, swapped episodes'))):
        x = np.arange(4)
        for i, n in enumerate(MODELS):
            m = [summary[n][f'{pre}/{a}']['all_seeds_mean'] for a in ax_names]
            ax.bar(x + (i - 1) * 0.27, m, 0.25, color=COL[n], label=n)
        ax.set_xticks(x)
        ax.set_xticklabels(['h1->self', 'h1->other', 'h2->self', 'h2->other'], fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.grid(axis='y', color='0.93')
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel('R^2 (all seeds mean, late-5)')
    fig.suptitle('v10 B4: 4-axis position R^2')
    fig.savefig(os.path.join(B4, 'b4_r2_bars.png'), dpi=110)
    plt.close(fig)

    vision_plot(summary)


def vision_plot(summary):
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    x = np.arange(len(MODELS))
    for i, sub in enumerate(('original', 'swap_rg')):
        m = [summary[n][f'vision_error/{sub}']['all_seeds_mean'] for n in MODELS]
        ax.bar(x + (i - 0.5) * 0.38, m, 0.35, color=[COL[n] for n in MODELS], hatch='' if i == 0 else '//',
               edgecolor='k', label=sub)
        for j, n in enumerate(MODELS):
            ax.text(x[j] + (i - 0.5) * 0.38, m[j] * 1.01, f'{m[j]:.4f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(MODELS))
    ax.set_ylabel('self-vision |pred - truth| per pixel (test)')
    ax.legend(fontsize=8)
    ax.set_title('v10 B4: self-vision prediction error by layout (all seeds mean)', fontsize=10)
    fig.savefig(os.path.join(B4, 'b4_vision_error.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
