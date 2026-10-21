"""
v12 stage 2c per-pair interim summary (docs/v12_experiment_log.md Sec.46): primary T2/T5 on the hidden test (main),
the visible-at-t50 subset and the visible test (references), and the priority secondary (B3 on the hidden test,
sign agreement, corr(Q2, P_A2) and T3 by hidden / visible frames). Reads the evaluator outputs only.
Usage: python analyze/v12/s2c_summary_v12.py --seeds 0 2
"""

import argparse
import json
import os

import numpy as np

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.path.join(_PI, 'data', 'result', 'v12')
EPS = [160, 170, 180, 190, 200]


def rng(xs):
    return f'{min(xs):+.3f}..{max(xs):+.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    a = ap.parse_args()
    for s in a.seeds:
        print(f'===== pair s{s}')
        for out, lab in (('stage2c', 'MAIN hidden'), ('stage2c_hidvis', 'ref visible-subset'), ('stage2c_vis', 'ref visible test')):
            agg = json.load(open(os.path.join(V, out, 'eval', f'aggregate_primary_interim_s{s}.json')))
            for key, nm in (('green:sT2_t50', 'T2'), ('pref:dT5', 'T5')):
                r = agg[key][f's{s}']
                pc = r['per_ckpt']
                print(f'  {lab:20s} {nm}: pass {r["n_pass"]}/5  mixed {rng([x["mixed"] for x in pc])}  '
                      f'random {rng([x["random"] for x in pc])}  diff {rng([x["diff"] for x in pc])}')
        # secondary from the main eval json (late-5 ranges)
        for st in ('green', 'pref'):
            for c in ('random', 'mixed'):
                js = [json.load(open(os.path.join(V, 'stage2c', 'eval', f'{st}_{c}_s{s}_ep{e}.json'))) for e in EPS]
                sa = js[-1]['sign_agreement']
                cf = [j['corr_Q2_PA2']['by_frames'] for j in js]
                t3 = [j['T3_self_L1_green_by_frames'] for j in js]
                print(f'  {st:5s} {c:6s} sign(ep200) hid {sa["green_hidden"]["original"]:.2f}/{sa["green_hidden"]["swap_rg"]:.2f} '
                      f'pref-hid {sa["pref_hidden"]["all"]:.2f} | corr hid {rng([x["green_hidden"] for x in cf])} vis '
                      f'{rng([x["green_visible"] for x in cf])} pref-hid {rng([x["pref_hidden"] for x in cf])} | '
                      f'T3 hid gen {np.mean([x["gen_hidden"] for x in t3]):.4f} zero {np.mean([x["zero_hidden"] for x in t3]):.4f} '
                      f'd6 {np.mean([x["d6_hidden"] for x in t3]):.4f}; vis gen {np.mean([x["gen_visible"] for x in t3]):.4f} '
                      f'd6 {np.mean([x["d6_visible"] for x in t3]):.4f}')
        for st in ('green', 'pref'):
            b = json.load(open(os.path.join(V, 'stage2c', 'b3', f'{st}_s{s}.json')))
            for k in ('green', 'pref'):
                for sub in ('hidden_t20', 'visible_t20'):
                    m = [b[f'mixed_ep{e}'][f'{k}_freeze'][sub]['g_other'] for e in EPS]
                    r = [b[f'random_ep{e}'][f'{k}_freeze'][sub]['g_other'] for e in EPS]
                    p = [b[f'paired_ep{e}'][f'{k}_freeze'][sub]['g_other'] for e in EPS]
                    ok = sum(mm['mean'] > 0 and mm['ci95'][0] > 0 and pp['ci95'][0] > 0 for mm, pp in zip(m, p))
                    print(f'  B3 G={st:5s} set={k:5s} {sub:11s}: mixed {rng([x["mean"] for x in m])} random '
                          f'{rng([x["mean"] for x in r])} paired {rng([x["mean"] for x in p])}  G5a-form points {ok}/5')


if __name__ == '__main__':
    main()
