"""
Paper round 4 (docs/paper_extra_log.md Sec.30; outside the pre-registration, rules and definitions committed before
computing): do the episodes whose A-2 position cannot be read from the predicted image, and are therefore dropped
from the means, bias the comparisons? Saved per-episode values only (CPU, no model is run).
  readable : G5b episode value finite (>= 1 of 8 bins with +a, -a and control read at t = 29); sign-following part
             = (G5a freeze g - flipped g) / 2 finite (both g finite). Plus frac_read_ctrl / G5a frac_read from the json.
  check    : the original verdict counts (robust6.json, C_signrand/summary.json, B_copy/summary.json, NS gate of v13
             Sec.28) and the 6-pair t tests (across_pairs.json) are reproduced first.
  variants : orig; a = episodes readable in every condition of the pair (per quantity); a3 = in none / 50% / 80%
             (6-pair tests only); b1 = unreadable -> 0 in both models of a comparison; b2 = unreadable -> 0 only in
             the model the hypothesis expects larger (the other side unchanged).
  bias     : paired bootstrap CI of the readable-fraction difference per point; properties of unreadable episodes.
-> data/result/paper_extra/R4/summary.json
"""

import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_PI, 'analyze', 'v13'), os.path.join(_PI, 'analyze', 'v11')):
    sys.path.insert(0, p)
from eval_trial_v11 import boot_ci  # noqa
from across_pairs_v13 import test as ttest  # noqa

RES = os.path.join(_PI, 'data', 'result')
DATA = os.path.join(_PI, 'data', 'data')
OUT = os.path.join(RES, 'paper_extra', 'R4')
EPS = (160, 170, 180, 190, 200)
CE = 'paper_extra/C_signrand/eval'
# model -> (g5 dir, key prefix, decomp dir, decomp name); dirs end before s<s>
M = {'none': ('v13/main/g5', 'random', 'v13/post/g5a_decomp', 'random'),
     '50': ('v13/v12ref/g5', 'mixed', 'v13/post/g5a_decomp', 'mixed50'),
     '80': ('v13/main/g5', 'mixed', 'v13/post/g5a_decomp', 'mixed80'),
     'NS': ('v13/ns/g5', 'mixed', 'v13/post/g5a_decomp_ns', 'mixed80'),
     'NSnone': ('v13/ns/g5', 'random', 'v13/post/g5a_decomp_ns', 'random'),
     'signrand': (f'{CE}/g5/signrand', 'signrand', f'{CE}/decomp/v13/post/g5a_decomp', 'signrand'),
     'pzero': (f'{CE}/g5/pzero', 'pzero', f'{CE}/decomp/v13/post/g5a_decomp', 'pzero'),
     'copy': ('paper_extra/B_copy/g5', 'mixed', 'paper_extra/B_copy/decomp/v13/post/g5a_decomp', 'mixed80'),
     'pcols': ('paper_extra/B_copy_pcols/g5', 'mixed', 'paper_extra/B_copy_pcols/decomp/v13/post/g5a_decomp', 'mixed80')}
ALL13 = tuple(M)
ALL46 = ('none', '50', '80')
QS = ('G5b', 'green', 'pref')
_cache = {}


def npz(path):
    if path not in _cache:
        z = np.load(os.path.join(RES, path))
        _cache[path] = {k: z[k] for k in z.files}
    return _cache[path]


def val(m, s, e, q):
    gd, key, dd, dn = M[m]
    g = npz(f'{gd}/s{s}/g5_per_ep{e}.npz')
    if q == 'G5b':
        return g[f'{key}__G5b']
    return (g[f'{key}__G5a_{q}_K15_freeze'] - npz(f'{dd}/s{s}/decomp_{dn}_ep{e}.npz')[f'{q}_flip_g']) / 2


def parts(m, s, e, q):
    """freeze g and flipped g of the sign-following part (for the readable fractions)."""
    gd, key, dd, dn = M[m]
    return npz(f'{gd}/s{s}/g5_per_ep{e}.npz')[f'{key}__G5a_{q}_K15_freeze'], npz(f'{dd}/s{s}/decomp_{dn}_ep{e}.npz')[f'{q}_flip_g']


def gjson(m, s, e):
    gd, key, _, _ = M[m]
    return json.load(open(os.path.join(RES, gd, f's{s}', f'g5_{key}_ep{e}.json')))


def models(s):
    return ALL13 if s <= 2 else ALL46


def summ(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def pos(r):
    return r['mean'] > 0 and r['ci95'][0] > 0


def neg(r):
    return r['mean'] < 0 and r['ci95'][1] < 0


# comparison -> (pairs, models used, model filled in b2, point rule on a dict of per-episode arrays)
COMPS = {
    '80_minus_none': ((0, 1, 2, 3, 4, 5), ('80', 'none'), '80', lambda x: {'exp': summ(x['80']), 'diff': summ(x['80'] - x['none'])},
                      lambda r: pos(r['exp']) and pos(r['diff'])),
    '50_minus_none': ((0, 1, 2, 3, 4, 5), ('50', 'none'), '50', lambda x: {'exp': summ(x['50']), 'diff': summ(x['50'] - x['none'])},
                      lambda r: pos(r['exp']) and pos(r['diff'])),
    '80_minus_50': ((0, 1, 2, 3, 4, 5), ('80', '50'), '80', lambda x: {'diff': summ(x['80'] - x['50'])}, lambda r: pos(r['diff'])),
    'signrand_minus_80': ((0, 1, 2), ('signrand', '80'), '80', lambda x: {'diff': summ(x['signrand'] - x['80'])}, lambda r: neg(r['diff'])),
    'pzero_minus_80': ((0, 1, 2), ('pzero', '80'), '80', lambda x: {'diff': summ(x['pzero'] - x['80'])}, lambda r: neg(r['diff'])),
    'signrand_minus_none': ((0, 1, 2), ('signrand', 'none'), 'signrand', lambda x: {'diff': summ(x['signrand'] - x['none'])},
                            lambda r: pos(r['diff'])),
    'NS_gate': ((0, 1, 2), ('NS', 'NSnone'), 'NS', lambda x: {'exp': summ(x['NS']), 'diff': summ(x['NS'] - x['NSnone'])},
                lambda r: pos(r['exp']) and pos(r['diff'])),
    'NS_minus_80': ((0, 1, 2), ('NS', '80'), '80', lambda x: {'diff': summ(x['NS'] - x['80'])}, lambda r: neg(r['diff'])),
    'B1_copy': ((0, 1, 2), ('copy', 'none', 'NS'), 'copy',
                lambda x: {'exp': summ(x['copy']), 'minus_none': summ(x['copy'] - x['none']), 'minus_NS': summ(x['copy'] - x['NS'])},
                lambda r: pos(r['exp']) and pos(r['minus_none']) and pos(r['minus_NS'])),
    'B3_pcols': ((0, 1, 2), ('pcols', 'none', 'NS'), 'pcols',
                 lambda x: {'exp': summ(x['pcols']), 'minus_none': summ(x['pcols'] - x['none']), 'minus_NS': summ(x['pcols'] - x['NS'])},
                 lambda r: pos(r['exp']) and pos(r['minus_none']) and pos(r['minus_NS'])),
}


def transform(X, variant, s, e, q, ms, fill):
    """X: model -> per-episode array (copy). Returns the arrays the variant evaluates."""
    X = {m: np.array(v, float) for m, v in X.items()}
    if variant in ('a', 'a3'):
        pool = models(s) if variant == 'a' else ALL46
        ok = np.all([np.isfinite(val(m, s, e, q)) for m in pool], 0)
        for m in X:
            X[m][~ok] = np.nan
    elif variant == 'b1':
        for m in ms:
            X[m][np.isnan(X[m])] = 0.0
    elif variant == 'b2':
        X[fill][np.isnan(X[fill])] = 0.0
    return X


def run_comparisons():
    out = {}
    for name, (pairs, ms, fill, rule, ok) in COMPS.items():
        for q in QS:
            for variant in ('orig', 'a', 'b1', 'b2'):
                per = {}
                for s in pairs:
                    pts = []
                    for e in EPS:
                        X = transform({m: val(m, s, e, q) for m in ms}, variant, s, e, q, ms, fill)
                        r = rule(X)
                        r['ok'] = bool(ok(r))
                        pts.append(r)
                    n = sum(p['ok'] for p in pts)
                    per[f's{s}'] = {'n_ok': n, 'pair_ok': n >= 3, 'points': pts}
                out[f'{name}|{q}|{variant}'] = {
                    'per_pair': per, 'pairs_ok_main3': sum(per[f's{s}']['pair_ok'] for s in pairs if s <= 2),
                    'pairs_ok_all': sum(v['pair_ok'] for v in per.values()), 'n_pairs': len(pairs)}
    return out


def run_across():
    out = {}
    for q in QS:
        for a, b in (('80', '50'), ('80', 'none'), ('50', 'none')):
            fill = a
            for variant in ('orig', 'a', 'a3', 'b1', 'b2'):
                diff = []
                for s in range(6):
                    va, vb = [], []
                    for e in EPS:
                        X = transform({a: val(a, s, e, q), b: val(b, s, e, q)}, variant, s, e, q, (a, b), fill)
                        va.append(np.nanmean(X[a])); vb.append(np.nanmean(X[b]))
                    diff.append(float(np.mean(va) - np.mean(vb)))
                out[f'{q}|{a}_minus_{b}|{variant}'] = {'all6': ttest(diff), 'main3': ttest(diff[:3]), 'per_pair_diff': diff}
    return out


def readable_tables():
    out = {}
    for s in range(6):
        for m in models(s):
            for e in EPS:
                d = {'G5b_episode': float(np.isfinite(val(m, s, e, 'G5b')).mean()),
                     'G5b_frac_read_ctrl': gjson(m, s, e)['G5b']['frac_read_ctrl']}
                for q in ('green', 'pref'):
                    fz, fl = parts(m, s, e, q)
                    d[f'{q}_freeze'] = float(np.isfinite(fz).mean())
                    d[f'{q}_flip'] = float(np.isfinite(fl).mean())
                    d[f'{q}_both'] = float((np.isfinite(fz) & np.isfinite(fl)).mean())
                    d[f'{q}_G5a_frac_read'] = gjson(m, s, e)['G5a'][f'{q}_K15_freeze']['frac_read']
                out[f'{m}|s{s}|ep{e}'] = d
    return out


def common_counts():
    out = {}
    for name, (pairs, ms, _, _, _) in COMPS.items():
        for q in QS:
            for s in pairs:
                out[f'{name}|{q}|s{s}'] = [int(np.all([np.isfinite(val(m, s, e, q)) for m in ms], 0).sum()) for e in EPS]
    for q in QS:
        for s in range(6):
            out[f'ALL|{q}|s{s}'] = [int(np.all([np.isfinite(val(m, s, e, q)) for m in models(s)], 0).sum()) for e in EPS]
            out[f'ALL3|{q}|s{s}'] = [int(np.all([np.isfinite(val(m, s, e, q)) for m in ALL46], 0).sum()) for e in EPS]
    return out


def readable_diffs():
    out = {}
    for name, (pairs, ms, _, _, _) in COMPS.items():
        a, b = ms[0], ms[1]
        for q in QS:
            for s in pairs:
                pts = []
                for e in EPS:
                    d = np.isfinite(val(a, s, e, q)).astype(float) - np.isfinite(val(b, s, e, q)).astype(float)
                    pts.append({'mean': float(d.mean()), 'ci95': boot_ci(d)})
                out[f'{a}_minus_{b}|{q}|s{s}'] = pts
    return out


def episode_props():
    """Properties of episodes unreadable at >= 1 of the 5 points vs readable at all 5, per model, pair and quantity."""
    out = {}
    g = h5py.File(os.path.join(DATA, 'v11_s1_random', 'data.h5'), 'r')['test']
    sp, op, lay = g['self_position'][()], g['other_position'][()], g['layout'][()]
    sets = {'G5b': (op[:, 20], sp[:, 29], lay, None, float(np.abs(op[:, 20:30] - op[:, 20:21]).max()))}
    for q, nm in (('green', 'v11_s2t_green'), ('pref', 'v11_s2_pref')):
        h = h5py.File(os.path.join(DATA, nm, 'data.h5'), 'r')['test']
        lk = h['a2_liked'][()] if 'a2_liked' in h else np.zeros(len(h['layout']), int)
        sets[q] = (h['other_position'][:, 20], h['self_position'][:, 20], h['layout'][()], lk, None)
    corners = np.array([[10, 10], [10, -10], [-10, 10], [-10, -10]], float)
    for q, (p2, p1, ly, lk, a2move) in sets.items():
        dc = np.min(np.linalg.norm(p2[:, None] - corners[None], axis=-1), 1)
        d12 = np.linalg.norm(p2 - p1, axis=1)
        out[f'_data|{q}'] = {'A2_to_corner_mean': float(dc.mean()), 'A1_A2_dist_mean': float(d12.mean()),
                             'layout0_frac': float((ly == 0).mean()), 'A2_max_move_t20_29': a2move}
        for s in range(6):
            bad = {}
            for m in models(s):
                miss = np.any([~np.isfinite(val(m, s, e, q)) for e in EPS], 0)
                bad[m] = miss
                if miss.sum() == 0:
                    continue
                r = {'n_unreadable_any': int(miss.sum()),
                     'n_unreadable_all5': int(np.all([~np.isfinite(val(m, s, e, q)) for e in EPS], 0).sum())}
                for k, v in (('A2_to_corner', dc), ('A1_A2_dist', d12), ('layout0', (ly == 0).astype(float))):
                    r[f'{k}_unreadable'] = float(v[miss].mean())
                    r[f'{k}_readable'] = float(v[~miss].mean())
                if lk is not None:
                    r['a2_likes_red_unreadable'] = float((lk[miss] == 1).mean())
                    r['a2_likes_red_readable'] = float((lk[~miss] == 1).mean())
                out[f'{m}|{q}|s{s}'] = r
            ms = list(bad)
            out[f'_overlap|{q}|s{s}'] = {f'{a}&{b}': [int((bad[a] & bad[b]).sum()), int((bad[a] | bad[b]).sum())]
                                          for i, a in enumerate(ms) for b in ms[i + 1:] if (bad[a] | bad[b]).any()}
    return out


def checks(comp, acr):
    """Original verdict counts and 6-pair tests vs the saved aggregates."""
    rb = json.load(open(os.path.join(RES, 'v13', 'extra', 'robust6.json')))
    ap = json.load(open(os.path.join(RES, 'v13', 'extra', 'across_pairs.json')))['tests']
    cs = json.load(open(os.path.join(RES, 'paper_extra', 'C_signrand', 'summary.json')))
    bc = json.load(open(os.path.join(RES, 'paper_extra', 'B_copy', 'summary.json')))['per_pair']
    qk = {'G5b': 'G5b', 'green': 'sign_green', 'pref': 'sign_pref'}
    ck = {'G5b': 'G5b', 'green': 'signfollow_green', 'pref': 'signfollow_pref'}
    bad = []
    for q in QS:
        for s in range(6):
            for name, key in (('80_minus_none', f'c_{qk[q]}_80'), ('50_minus_none', f'c_{qk[q]}_50'), ('80_minus_50', f'b_{qk[q]}')):
                if comp[f'{name}|{q}|orig']['per_pair'][f's{s}']['n_ok'] != rb[f's{s}'][key]:
                    bad.append((name, q, s))
        for X, name in (('signrand', 'signrand_minus_80'), ('pzero', 'pzero_minus_80'), ('signrand', 'signrand_minus_none')):
            nm = 'vs80' if '80' in name else 'vsnone'
            for s in range(3):
                if comp[f'{name}|{q}|orig']['per_pair'][f's{s}']['n_ok'] != cs[X]['per_pair'][f's{s}'][f'{ck[q]}_{nm}']['n_ok']:
                    bad.append((name, q, s))
        for a, b in (('80', '50'), ('80', 'none'), ('50', 'none')):
            k = {'G5b': 'G5b', 'green': 'sign_green', 'pref': 'sign_pref'}[q]
            if not np.allclose(acr[f'{q}|{a}_minus_{b}|orig']['per_pair_diff'], ap[f'{k}_{a}_minus_{b}']['per_pair_diff'], atol=1e-12):
                bad.append(('across', q, a, b))
    for s in range(3):
        for name, key in (('B1_copy', 'B1'), ('B3_pcols', 'B3')):
            if comp[f'{name}|G5b|orig']['per_pair'][f's{s}']['n_ok'] != bc[f's{s}'][key]['n_ok']:
                bad.append((name, s))
    ns = [comp[f'NS_gate|G5b|orig']['per_pair'][f's{s}']['n_ok'] for s in range(3)]
    return {'mismatches': [list(map(str, b)) for b in bad], 'NS_gate_G5b_points': ns}


def main():
    os.makedirs(OUT, exist_ok=True)
    comp, acr = run_comparisons(), run_across()
    res = {'check': checks(comp, acr), 'comparisons': comp, 'across_pairs': acr, 'readable': readable_tables(),
           'common_counts': common_counts(), 'readable_diff': readable_diffs(), 'episode_props': episode_props()}
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print('check', json.dumps(res['check']))


if __name__ == '__main__':
    main()
