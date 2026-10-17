"""
v11 self-side gate: aggregation, pre-registered judgement and figures
(docs/v11_experiment_log.md Sec.7.13 criteria, Sec.7.18 items 3-10, 19).

Inputs per (seed, checkpoint): eval_m1_v11.py JSON (M1(a'), M1(b'), M1(c),
M1(d), M2, R^2) and m1d_masked_v11.py JSON (M1(d) 'none' / 'rg_dil1' and the
colour criterion S_Red - S_Green in swap episodes), optionally the
convergence JSON (check_convergence_v11.py).

Judgement (Sec.7.13):
  per seed pair and checkpoint:
    M1(c) ok  <=> mixed c_pred < 0  and paired diff CI upper < 0
    M1(d) ok  <=> mixed c_cf > 0    and paired diff CI lower > 0
    colour ok <=> mixed (S_Red - S_Green | swap episodes) > 0 with CI lower > 0
  a seed (pair) passes if ok at >= 3 of its 5 late checkpoints;
  a criterion passes if >= 2 of 3 seeds pass.
  "value-action" = M1(c) and M1(d) both pass; "value-colour" = colour passes.
Report: every seed individually (late-5 mean +- sd over checkpoints), the
all-seed mean +- sd, then the converged-seed mean +- sd.
Writes JSON + figures under data/result/v11/<sub>/.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V11 = os.path.join(_PI_DIR, 'data', 'result', 'v11')


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m1_fmt', required=True, help='path under data/result/v11 with {seed} {epoch}')
    ap.add_argument('--m1d_fmt', required=True, help='same for m1d_masked JSONs')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--convergence', default=None, help='convergence JSON under data/result/v11')
    ap.add_argument('--min_ckpt', type=int, default=3)
    ap.add_argument('--min_seed', type=int, default=2)
    ap.add_argument('--sub', default='main/gate')
    ap.add_argument('--label', default='main')
    args = ap.parse_args()
    out = os.path.join(V11, args.sub)
    os.makedirs(out, exist_ok=True)
    conv = load(os.path.join(V11, args.convergence)) if args.convergence else {}

    rows = {}   # seed -> metric -> list over epochs
    oks = {}
    for s in args.seeds:
        R = {k: [] for k in ('c_pred_mixed', 'c_pred_random', 'c_pred_diff', 'c_cf_mixed', 'c_cf_random', 'c_cf_diff',
                             'c_cf_masked_mixed', 'c_cf_masked_random', 'c_cf_masked_diff', 'colour_mixed', 'colour_random',
                             'colour_masked_mixed', 'a_prime_mixed', 'a_prime_random', 'a_prime_diff',
                             'b_prime_mixed', 'b_prime_random', 'b_prime_diff', 'r2_mixed', 'r2_random',
                             'M2prop_mixed')}
        ok = {'M1c': [], 'M1d': [], 'colour': []}
        for e in args.epochs:
            m = load(os.path.join(V11, args.m1_fmt.format(seed=s, epoch=e)))
            q = load(os.path.join(V11, args.m1d_fmt.format(seed=s, epoch=e)))
            P = m['paired_mixed_minus_random']
            g = lambda d: (d['mean'], d['ci95'][0], d['ci95'][1])  # noqa
            R['c_pred_mixed'].append(g(m['mixed']['M1c']['c_pred'])); R['c_pred_random'].append(g(m['random']['M1c']['c_pred']))
            R['c_pred_diff'].append(g(P['M1c_c_pred']))
            R['c_cf_mixed'].append(g(q['mixed']['none']['c_cf'])); R['c_cf_random'].append(g(q['random']['none']['c_cf']))
            R['c_cf_diff'].append(g(q['paired_mixed_minus_random']['none']))
            R['c_cf_masked_mixed'].append(g(q['mixed']['rg_dil1']['c_cf'])); R['c_cf_masked_random'].append(g(q['random']['rg_dil1']['c_cf']))
            R['c_cf_masked_diff'].append(g(q['paired_mixed_minus_random']['rg_dil1']))
            R['colour_mixed'].append(g(q['mixed']['none']['colour_swap_Sred_minus_Sgreen']))
            R['colour_random'].append(g(q['random']['none']['colour_swap_Sred_minus_Sgreen']))
            R['colour_masked_mixed'].append(g(q['mixed']['rg_dil1']['colour_swap_Sred_minus_Sgreen']))
            R['a_prime_mixed'].append(g(m['mixed']['M1a_prime']['c_red_specific'])); R['a_prime_random'].append(g(m['random']['M1a_prime']['c_red_specific']))
            R['a_prime_diff'].append(g(P['M1a_prime_c']))
            R['b_prime_mixed'].append(g(m['mixed']['M1b_prime']['main_mean8'])); R['b_prime_random'].append(g(m['random']['M1b_prime']['main_mean8']))
            R['b_prime_diff'].append(g(P['M1b_prime_main']))
            R['r2_mixed'].append((m['mixed']['r2_v4conv_insample_own_train'],) * 3); R['r2_random'].append((m['random']['r2_v4conv_insample_own_train'],) * 3)
            R['M2prop_mixed'].append(g(m['mixed']['M2_proposal_swap_newRed_minus_oldPlace']))
            ok['M1c'].append(R['c_pred_mixed'][-1][0] < 0 and R['c_pred_diff'][-1][2] < 0)
            ok['M1d'].append(R['c_cf_mixed'][-1][0] > 0 and R['c_cf_diff'][-1][1] > 0)
            ok['colour'].append(R['colour_mixed'][-1][0] > 0 and R['colour_mixed'][-1][1] > 0)
        rows[s] = R
        oks[s] = {k: {'per_ckpt': v, 'n_ok': int(sum(v)), 'seed_pass': bool(sum(v) >= args.min_ckpt)} for k, v in ok.items()}

    judge = {k: {'seeds_pass': int(sum(oks[s][k]['seed_pass'] for s in args.seeds)),
                 'pass': bool(sum(oks[s][k]['seed_pass'] for s in args.seeds) >= args.min_seed)}
             for k in ('M1c', 'M1d', 'colour')}
    judge['value_action'] = bool(judge['M1c']['pass'] and judge['M1d']['pass'])
    judge['value_colour'] = bool(judge['colour']['pass'])

    def conv_ok(s):
        r = conv.get(f'mixed_s{s}', {}).get('self_vision_per_pixel', {})
        r2 = conv.get(f'random_s{s}', {}).get('self_vision_per_pixel', {})
        return bool(r.get('converged') and r2.get('converged'))

    table = {}
    for k in rows[args.seeds[0]]:
        per_seed = {s: float(np.mean([x[0] for x in rows[s][k]])) for s in args.seeds}
        per_seed_sd = {s: float(np.std([x[0] for x in rows[s][k]])) for s in args.seeds}
        vals = np.array(list(per_seed.values()))
        convs = [s for s in args.seeds if conv_ok(s)] if conv else []
        table[k] = {'per_seed_late5_mean': per_seed, 'per_seed_ckpt_sd': per_seed_sd,
                    'all_seed_mean': float(vals.mean()), 'all_seed_sd': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    'converged_seeds': convs,
                    'converged_mean': float(np.mean([per_seed[s] for s in convs])) if convs else None,
                    'converged_sd': float(np.std([per_seed[s] for s in convs], ddof=1)) if len(convs) > 1 else None}
    res = {'label': args.label, 'epochs': args.epochs, 'seeds': args.seeds, 'judgement': judge, 'per_seed_ok': oks,
           'table': table, 'raw': {str(s): rows[s] for s in args.seeds}}
    with open(os.path.join(out, f'gate_{args.label}.json'), 'w') as f:
        json.dump(res, f, indent=1)

    # figure: main criteria + secondary, per seed bars (late-5 mean, whiskers = mean of the per-ckpt CI bounds)
    panels = [('M1(c) c_pred (main; <0)', 'c_pred'), ('M1(d) c_cf (main; >0)', 'c_cf'),
              ('M1(d) red/green-excluded', 'c_cf_masked'), ('colour: S_red - S_green, swap eps (main; >0)', 'colour'),
              ("M1(a') (secondary)", 'a_prime'), ("M1(b') (secondary)", 'b_prime')]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
    for ax, (title, key) in zip(axes, panels):
        xs, labels = 0, []
        for s in args.seeds:
            for cond, col in (('random', 'tab:gray'), ('mixed', 'tab:red')):
                k = f'{key}_{cond}'
                if k not in rows[s]:
                    continue
                arr = np.array(rows[s][k])
                mu, lo, hi = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
                ax.bar(xs, mu, color=col, yerr=[[mu - lo], [hi - mu]], capsize=3)
                labels.append(f's{s}\n{cond[0]}')
                xs += 1
            xs += 0.5
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks([]); ax.set_title(title, fontsize=9)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color='tab:gray'), plt.Rectangle((0, 0), 1, 1, color='tab:red')],
                   labels=['random (no experience)', 'mixed (experience)'], fontsize=7)
    fig.suptitle(f'v11 self-side gate ({args.label}): per seed, mean over checkpoints {args.epochs}; whiskers = mean 95% CI bounds. '
                 f"Judgement: value-action {judge['value_action']} (M1c {judge['M1c']['seeds_pass']}/{len(args.seeds)}, "
                 f"M1d {judge['M1d']['seeds_pass']}/{len(args.seeds)}), value-colour {judge['value_colour']} "
                 f"({judge['colour']['seeds_pass']}/{len(args.seeds)})", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'gate_{args.label}.png'), dpi=120)
    print(json.dumps(judge, indent=1))


if __name__ == '__main__':
    main()
