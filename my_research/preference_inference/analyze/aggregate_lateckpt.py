"""
Aggregate the last-5-checkpoint evaluation into mean +/- sd (inter-checkpoint
variation = training-oscillation effect on the result). research_plan Sec.P0-5.

Reads the per-checkpoint JSONs that run_eval_v4v5.sh wrote to
data/result/baseline_v4/ for epochs {E, E-10, E-20, E-30, E-40}:
  <EXP>_s<SEED>_ep<EP>_<DATASET>_es0_r2.json          (4-axis R^2)
  <EXP>_s<SEED>_ep<EP>_<DATASET>_es0_q2pos_q2pos.json  (Q2hat -> pos)     [VE models]
  <EXP>_s<SEED>_ep<EP>_<DATASET>_es0_r4ve.json          (real/zero/true_q2/...) [VE models]

Prints mean, sd, and (for the ablation) whether |mean(true_q2 - zero)| exceeds
the inter-checkpoint sd -- i.e. whether that claim survives training oscillation.

Usage:
  python analyze/aggregate_lateckpt.py <EXP> <E> --seed 0 --dataset r2_a1random_a2rl
"""
import argparse
import glob
import json
import os
import statistics as st

BV4 = 'data/result/baseline_v4'


def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = st.mean(xs)
    s = st.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s, len(xs)


def _load(pattern):
    fs = sorted(glob.glob(os.path.join(BV4, pattern)))
    return [json.load(open(f)) for f in fs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp')
    ap.add_argument('E', type=int)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dataset', default='r2_a1random_a2rl')
    args = ap.parse_args()

    eps = [args.E - k for k in (0, 10, 20, 30, 40)]
    tag = f'{args.exp}_s{args.seed}_ep%d_{args.dataset}_es0'

    print(f'=== late-5-checkpoint aggregate: {args.exp} seed{args.seed}  '
          f'epochs {eps} ===')

    # 4-axis R^2
    r2 = {k: [] for k in ('h1_to_self', 'h1_to_other', 'h2_to_self', 'h2_to_other')}
    for ep in eps:
        fs = glob.glob(os.path.join(BV4, (tag % ep) + '_r2.json'))
        if not fs:
            continue
        d = json.load(open(fs[0]))
        for k in r2:
            r2[k].append(d.get(k))
    print('\n4-axis R^2 (mean +/- sd over checkpoints):')
    for k, v in r2.items():
        m, s, n = _mean_sd(v)
        if m is not None:
            print(f'  {k:14s}: {m:.4f} +/- {s:.5f}  (n={n})  raw={[round(x,4) for x in v]}')

    # r4_ve ablation
    ab = {k: [] for k in ('real', 'zero', 'true_q2', 'constant_mean')}
    for ep in eps:
        fs = glob.glob(os.path.join(BV4, (tag % ep) + '_r4ve.json'))
        if not fs:
            continue
        d = json.load(open(fs[0]))
        v = d.get('vision_l1_by_ablation', {})
        for k in ab:
            ab[k].append(v.get(k))
    if any(ab['real']):
        print('\nr4_ve_eval ablation (mean +/- sd over checkpoints):')
        for k, v in ab.items():
            m, s, n = _mean_sd(v)
            print(f'  {k:14s}: {m:.5f} +/- {s:.5f}  (n={n})  raw={[round(x,5) for x in v]}')
        tqz = [t - z for t, z in zip(ab['true_q2'], ab['zero']) if t is not None and z is not None]
        rz = [r - z for r, z in zip(ab['real'], ab['zero']) if r is not None and z is not None]
        for name, d in (('true_q2 - zero', tqz), ('real - zero', rz)):
            m, s, n = _mean_sd(d)
            verdict = ('SURVIVES (|mean| > sd)' if abs(m) > s
                       else 'FRAGILE (|mean| <= inter-ckpt sd)')
            print(f'  Δ {name:16s}: mean={m:+.5f}  inter-ckpt sd={s:.5f}  n={n}  -> {verdict}')

    # q2 -> position
    qp = {k: [] for k in ('q2_to_self_pos_r2', 'q2_to_other_pos_r2')}
    for ep in eps:
        fs = glob.glob(os.path.join(BV4, (tag % ep) + '_q2pos*q2pos.json')) \
            or glob.glob(os.path.join(BV4, (tag % ep) + '_q2pos.json'))
        if not fs:
            continue
        d = json.load(open(fs[0]))
        for k in qp:
            qp[k].append(d.get(k))
    if any(qp['q2_to_self_pos_r2']):
        print('\nQ2hat -> position R^2 (mean +/- sd over checkpoints):')
        for k, v in qp.items():
            m, s, n = _mean_sd(v)
            print(f'  {k:22s}: {m:.4f} +/- {s:.5f}  (n={n})  raw={[round(x,4) for x in v]}')

    out = os.path.join(BV4, f'{args.exp}_s{args.seed}_late5_aggregate.json')
    json.dump({'exp': args.exp, 'seed': args.seed, 'epochs': eps,
               'r2': r2, 'ablation': ab, 'q2pos': qp}, open(out, 'w'), indent=2)
    print(f'\nsaved: {out}')


if __name__ == '__main__':
    main()
