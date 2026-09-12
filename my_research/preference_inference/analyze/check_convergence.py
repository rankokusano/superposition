"""
Threshold-based convergence check (research_plan_v4_v5.md Sec.P0-5, finalised
2026-09-06).

Target metric: feature_prediction_other from the TRAIN log (one value per epoch).
  E = total epochs, W = final 20% of epochs (>= 30), B = post-burn-in (epoch >= ceil(0.1E)).
  m_W    = mean(fp_other over W)
  v_min  = min(fp_other over B)          <- absolute level, reported separately
          (discriminates "settled low" vs "settled on a high plateau / never transitioned")
  slope_W = OLS slope of fp_other over W

CONVERGED iff BOTH:
  (i)  m_W <= 1.25 * v_min                              (settled near its own best)
  (ii) |slope_W| * |W| <= 0.10 * v_min                  (net drift over final window < 10% of best)

Also reported per run: transition epoch = first epoch with fp_other < 1.3 * v_min
(or "none").

Usage (from /work/my_research/preference_inference):
  python analyze/check_convergence.py <exp_config> [<exp_config> ...] [--seeds 0 1 2]
  python analyze/check_convergence.py r3_a_direct_1000pretrain_400 --seeds 0 1 2
"""
import argparse
import glob
import json
import os

LOCAL = 'data/result'
ROOT = '/work/data/result'


def find_log(exp, seed):
    for base in (LOCAL, ROOT):
        p = os.path.join(base, exp, str(seed), 'log', 'train',
                         'feature_prediction_other.log')
        if os.path.isfile(p):
            return p
    return None


def load_series(path):
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep, val = line.split(',')
            xs.append(int(ep))
            ys.append(float(val))
    return xs, ys


def ols_slope(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    return num / den if den else 0.0


def check(xs, ys):
    E = len(ys)
    w_len = max(30, int(round(0.2 * E)))
    w_len = min(w_len, E)
    burn = max(0, int(-(-E // 10)))  # ceil(0.1E)
    B = ys[burn:] if burn < E else ys
    W_x = xs[E - w_len:]
    W_y = ys[E - w_len:]
    v_min = min(B)
    m_W = sum(W_y) / len(W_y)
    slope = ols_slope(W_x, W_y)
    drift = abs(slope) * (W_x[-1] - W_x[0])
    cond_i = m_W <= 1.25 * v_min
    cond_ii = drift <= 0.10 * v_min
    converged = cond_i and cond_ii
    thr = 1.3 * v_min
    trans = next((xs[i] for i, v in enumerate(ys) if v < thr), None)
    return dict(
        total_epochs=E, window=w_len, burn_in_epoch=xs[burn] if burn < E else xs[0],
        v_min=round(v_min, 3), m_W=round(m_W, 3),
        slope_per_epoch=round(slope, 5), drift_over_window=round(drift, 3),
        cond_i_level=cond_i, cond_ii_no_rise=cond_ii,
        CONVERGED=converged,
        transition_epoch=(trans if trans is not None else 'none'),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp_configs', nargs='+')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--out', default='data/result/baseline_v4/convergence_check.json')
    args = ap.parse_args()

    results = {}
    for exp in args.exp_configs:
        for s in args.seeds:
            path = find_log(exp, s)
            if path is None:
                continue
            xs, ys = load_series(path)
            if not ys:
                continue
            r = check(xs, ys)
            r['log'] = path
            key = f'{exp}/seed{s}'
            results[key] = r
            print(f'{key:42s} E={r["total_epochs"]:>4} '
                  f'v_min={r["v_min"]:>7} m_W={r["m_W"]:>7} '
                  f'drift={r["drift_over_window"]:>7} '
                  f'(i){"OK " if r["cond_i_level"] else "NG "} '
                  f'(ii){"OK " if r["cond_ii_no_rise"] else "NG "} '
                  f'-> {"CONVERGED" if r["CONVERGED"] else "NOT-converged"}  '
                  f'trans@{r["transition_epoch"]}')

    # convergence rate per config
    print()
    by_cfg = {}
    for k, v in results.items():
        cfg = k.split('/seed')[0]
        by_cfg.setdefault(cfg, []).append(v['CONVERGED'])
    for cfg, flags in by_cfg.items():
        print(f'{cfg:42s} convergence rate: {sum(flags)}/{len(flags)}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved: {args.out}')


if __name__ == '__main__':
    main()
