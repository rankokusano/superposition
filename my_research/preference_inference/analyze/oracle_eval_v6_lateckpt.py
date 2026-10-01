"""
S4 late-5 aggregation (2026-10-01, user request): oracle_eval_v6.py's
single-epoch run only checked S3's final checkpoint (ep200). The project's
own convention (never judge by a single checkpoint) and the user's direct
request both call for aggregating across the late-5 checkpoints
(200,190,180,170,160) -- same set S2/S3's regression_baseline_v4.py runs
used -- to get an inter-checkpoint mean +/- sd for:
  - overall vision_l1 per mode
  - the SS28.2 distance-binned vision_l1 AND h2->other R^2 per mode
This directly answers whether the >=15 bin's true_r effect (n=5573, 1.9%
of frames) is stable across nearby checkpoints or within noise, and
whether 'constant' shows the same distance trend as 'true_r' (if it does,
the SS28.2 finding would be "any nonzero signal helps at range", not
"the reward information helps at range").

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/oracle_eval_v6_lateckpt.py --exp_config v6_s3_base_l1 --last_epoch 200
"""
import argparse
import json
import os

import numpy as np

from oracle_eval_v6 import run_epoch, MODES, BIN_LABELS, SAVE_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', default='v6_s3_base_l1')
    ap.add_argument('--last_epoch', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_episodes', type=int, default=3000)
    ap.add_argument('--label', default=None)
    args = ap.parse_args()
    label = args.label or f'{args.exp_config}_oracle_lateckpt'
    epochs = [args.last_epoch - 10 * k for k in range(5)]

    per_epoch = []
    for ep in epochs:
        print(f'\n########## epoch {ep} ##########')
        per_epoch.append(run_epoch(args.exp_config, args.seed, ep, args.n_episodes, verbose=True))

    def agg(vals):
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    overall = {}
    print('\n=== Aggregate (mean +/- inter-checkpoint sd over epochs %s) ===' % epochs)
    print('-- overall self_vision L1 (ov_enc=other_vision) --')
    for mode in MODES:
        vals = [r['vision_l1_by_condition'][f'other_vision__{mode}'] for r in per_epoch]
        m, sd = agg(vals)
        overall[mode] = {'mean': m, 'sd': sd, 'raw': vals}
        print(f'  {mode:10s}: {m:.4f} +/- {sd:.4f}')
    d_vals = [r['true_r_minus_zero']['other_vision'] for r in per_epoch]
    dm, dsd = agg(d_vals)
    print(f'  true_r - zero: {dm:+.4f} +/- {dsd:.4f}')

    dist_agg = {}
    print('\n-- distance-binned self_vision L1 (mean +/- inter-checkpoint sd) --')
    for lbl in BIN_LABELS:
        n_vals = [r['distance_binned_ablation'][lbl]['n'] for r in per_epoch]
        row = {}
        print(f'  {lbl:6s} (n~{int(np.mean(n_vals))}):')
        for mode in MODES:
            vals = [r['distance_binned_ablation'][lbl]['vision_l1'][mode] for r in per_epoch]
            m, sd = agg(vals)
            row[mode] = {'mean': m, 'sd': sd, 'raw': vals}
            print(f'    {mode:10s}: {m:.4f} +/- {sd:.4f}')
        delta_vals = [r['distance_binned_ablation'][lbl]['vision_l1']['true_r'] -
                      r['distance_binned_ablation'][lbl]['vision_l1']['zero'] for r in per_epoch]
        dm2, dsd2 = agg(delta_vals)
        row['true_r_minus_zero'] = {'mean': dm2, 'sd': dsd2, 'raw': delta_vals}
        const_delta_vals = [r['distance_binned_ablation'][lbl]['vision_l1']['constant'] -
                             r['distance_binned_ablation'][lbl]['vision_l1']['zero'] for r in per_epoch]
        cm2, csd2 = agg(const_delta_vals)
        row['constant_minus_zero'] = {'mean': cm2, 'sd': csd2, 'raw': const_delta_vals}
        print(f'    true_r-zero : {dm2:+.4f} +/- {dsd2:.4f}')
        print(f'    constant-zero: {cm2:+.4f} +/- {csd2:.4f}')
        dist_agg[lbl] = {'n_mean': float(np.mean(n_vals)), **row}

    dist_h2_agg = {}
    print('\n-- distance-binned h2->other R^2 (mean +/- inter-checkpoint sd) --')
    for lbl in BIN_LABELS:
        row = {}
        print(f'  {lbl:6s}:')
        for mode in MODES:
            vals = [r['distance_binned_ablation'][lbl]['h2_to_other'][mode] for r in per_epoch]
            m, sd = agg(vals)
            row[mode] = {'mean': m, 'sd': sd, 'raw': vals}
            print(f'    {mode:10s}: {m:.4f} +/- {sd:.4f}')
        dist_h2_agg[lbl] = row

    h2_agg = {}
    print('\n-- overall h2->self / h2->other R^2 (mean +/- inter-checkpoint sd) --')
    for mode in MODES:
        self_vals = [r['h2_r2_by_mode'][mode]['h2_to_self'] for r in per_epoch]
        other_vals = [r['h2_r2_by_mode'][mode]['h2_to_other'] for r in per_epoch]
        sm, ssd = agg(self_vals)
        om, osd = agg(other_vals)
        h2_agg[mode] = {'h2_to_self': {'mean': sm, 'sd': ssd}, 'h2_to_other': {'mean': om, 'sd': osd}}
        print(f'  {mode:10s}: h2->self={sm:.4f}+/-{ssd:.4f}  h2->other={om:.4f}+/-{osd:.4f}')

    result = {
        'label': label, 'exp_config': args.exp_config, 'epochs': epochs,
        'overall_vision_l1': overall,
        'distance_binned_vision_l1': dist_agg,
        'distance_binned_h2_to_other': dist_h2_agg,
        'overall_h2_r2': h2_agg,
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{label}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
