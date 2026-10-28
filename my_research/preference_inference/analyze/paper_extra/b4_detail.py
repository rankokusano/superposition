import json, sys, numpy as np
sys.path.insert(0, 'analyze/paper_extra')
import c_aggregate as ca
R = json.load(open('data/result/paper_extra/B4/summary.json'))
C = json.load(open('data/result/paper_extra/C_signrand/summary.json'))['signrand']['per_pair']
f = lambda p: f"{p['mean']:+.3f}[{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}]"
out = {}
for s in range(3):
    P = R['per_pair'][f's{s}']
    print(f'== pair {s+1}')
    for q in ('G5b', 'signfollow_green', 'signfollow_pref'):
        for nm in ('sr_vs80', 'sr_vsnone', '80_vsnone'):
            v = P[f'{q}_{nm}']
            print(f'  B4 {q:17s} {nm:9s} n_ok {v["n_ok"]} late5 {np.mean([p["mean"] for p in v["points"]]):+.3f} |', ' '.join(f(p) for p in v['points']))
    fn = {'G5b': lambda c, e: ca.g5(c, s, e, 'G5b'), 'signfollow_green': lambda c, e: ca.sf(c, s, e, 'green'),
          'signfollow_pref': lambda c, e: ca.sf(c, s, e, 'pref')}
    for q in fn:
        pw = ca.pointwise(fn[q], 'mixed80', 'random', 1)
        out[f's{s}_{q}_80_vsnone_common'] = pw
        for nm, key in (('sr_vs80', f'{q}_vs80'), ('sr_vsnone', f'{q}_vsnone')):
            v = C[f's{s}'][key]
            print(f'  CT {q:17s} {nm:9s} n_ok {v["n_ok"]} late5 {np.mean([p["mean"] for p in v["points"]]):+.3f}')
        print(f'  CT {q:17s} 80_vsnone n_ok {pw["n_ok"]} late5 {np.mean([p["mean"] for p in pw["points"]]):+.3f}')
    for c in ('signrand', 'mixed80', 'random'):
        d = P['values'][c]
        print(f'  {c:9s} G5a_frac_read', {k: [round(x, 3) for x in v] for k, v in d['G5a_frac_read'].items()}, 'frac_read_ctrl', [round(x, 3) for x in d['frac_read_ctrl']])
        print(f'  {c:9s} track vis', round(min(d['track_visible']), 3), round(max(d['track_visible']), 3), 'hid', round(min(d['track_hidden']), 3), round(max(d['track_hidden']), 3), '| common', {q: round(float(np.mean(v)), 3) for q, v in d['common_test'].items()})
json.dump(out, open('data/result/paper_extra/B4/common_80_vs_none.json', 'w'), indent=1)
