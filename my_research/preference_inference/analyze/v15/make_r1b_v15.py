"""
v15 Sec.39 R1b (pre-registered before the R1 / R2 results): start checkpoint = R1 (make_r12_v15.py) with the other side's
11 input columns multiplied by ONE constant per seed, c_s = ||shared trained lstm.weight_ih[:, 64:]|| / ||NS
lstm_other.weight_ih[:, 64:]|| (Frobenius, whole 11-column block), so the block keeps the NS values' pattern (initial values
after weight decay) at the shared trained size. -> data/result/v15_s2p1_noexp_v15i_r1b/<seed>/model/00150.pth
Checks (CPU, stop on failure): every tensor except the 11 columns equals R1 exactly; the 11 columns equal c_s x NS; their
norm equals the shared norm (relative 1e-5: float32 rounding of the scaled
block is ~3e-6); loaded as in train_v12.py the state equals the checkpoint and G (initial
values) equals the shared phase-2 ep0. Per-column norms are written for description.

    python analyze/v15/make_r1b_v15.py   # -> data/result/v15/stage2/make_r1b_v15.json
"""
import json
import os
import sys

import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from analyze.v15.make_r12_v15 import NV, P, WI, ck, load_like_training, save, suf  # noqa: E402
from analyze.v15 import eval_s2_v15 as E  # noqa: E402


def main():
    torch.set_num_threads(2)
    R, ok = {}, True
    for s in (0, 1, 2):
        sh = ck('v15_s2p1_noexp_v15i_mse', s, 150)[P + 'lstm.weight_ih'][:, NV:]
        ns = ck('v15_s2p1_noexp_v15i_ns_mse', s, 150)[WI][:, NV:]
        r1 = ck('v15_s2p1_noexp_v15i_r1', s, 150)
        c = float(sh.norm() / ns.norm())
        r1b = {k: v.clone() for k, v in r1.items()}
        r1b[WI][:, NV:] = ns * c
        save('r1b', s, r1b)
        r = {'c': c,
             'rest_equal_to_r1': all(torch.equal(r1b[k], r1[k]) for k in r1 if k != WI) and torch.equal(r1b[WI][:, :NV], r1[WI][:, :NV]),
             'cols_equal_c_times_ns': bool(torch.equal(r1b[WI][:, NV:], ns * c)),
             'norm': {'shared': float(sh.norm()), 'ns': float(ns.norm()), 'r1b': float(r1b[WI][:, NV:].norm())},
             'col_norms': {'shared': sh.norm(dim=0).tolist(), 'ns': ns.norm(dim=0).tolist(),
                           'r1b': r1b[WI][:, NV:].norm(dim=0).tolist()},
             'cos_r1b_shared': float((r1b[WI][:, NV:] * sh).sum() / (r1b[WI][:, NV:].norm() * sh.norm()))}
        r['norm_matches'] = abs(r['norm']['r1b'] / r['norm']['shared'] - 1) < 1e-5
        m = load_like_training(f'v15_s2p2h_noexp_v15i_r1b{suf(s)}', 'r1b', s)
        st, rs = m.state_dict(), E.load(f'v15_s2p2h_noexp_v15i{suf(s)}', 0, torch.device('cpu'), seed=s).state_dict()
        rest = [k for k in st if k not in r1b]
        r['loaded_equals_ckpt'] = all(torch.equal(st[k], r1b[k]) for k in r1b)
        r['rest_is_G_equal_shared_ep0'] = all(k.startswith('motion_generator_module.') for k in rest) and \
            all(torch.equal(st[k], rs[k]) for k in rest)
        r['passed'] = all(v for v in r.values() if isinstance(v, bool))
        ok &= r['passed']
        R[s] = r
        print(s, {k: v for k, v in r.items() if k != 'col_norms'}, flush=True)
    R['all_passed'] = ok
    json.dump(R, open(os.path.join(E.OUT, 'make_r1b_v15.json'), 'w'), indent=1)
    print('ALL_PASSED' if ok else 'FAILED', flush=True)


if __name__ == '__main__':
    main()
