"""
v15 Sec.41 R2b (pre-registered before the R1b / R2b results): start checkpoint = the same-seed NS noexp phase 1 (ep150) with the
other side's 11 input columns multiplied by the R1b constant c_s = ||shared trained lstm.weight_ih[:, 64:]|| / ||NS
lstm_other.weight_ih[:, 64:]|| (one constant for the whole 11-column block). Its 11 columns are therefore identical to R1b's;
R1b and R2b differ only in the rest of the other-side LSTM (shared trained vs NS). -> data/result/v15_s2p1_noexp_v15i_r2b/<seed>/model/00150.pth
Checks (CPU, stop on failure): every tensor except the 11 columns equals NS exactly; the 11 columns equal c_s x NS and equal
R1b's; their norm equals the shared norm (relative 1e-5, as R1b); loaded as in train_v12.py the state equals the checkpoint and
G (initial values) equals the shared phase-2 ep0.

    python analyze/v15/make_r2b_v15.py   # -> data/result/v15/stage2/make_r2b_v15.json
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
        nsd = ck('v15_s2p1_noexp_v15i_ns_mse', s, 150)
        ns = nsd[WI][:, NV:]
        r1b = ck('v15_s2p1_noexp_v15i_r1b', s, 150)
        c = float(sh.norm() / ns.norm())
        r2b = {k: v.clone() for k, v in nsd.items()}
        r2b[WI][:, NV:] = ns * c
        save('r2b', s, r2b)
        r = {'c': c,
             'rest_equal_to_ns': all(torch.equal(r2b[k], nsd[k]) for k in nsd if k != WI) and torch.equal(r2b[WI][:, :NV], nsd[WI][:, :NV]),
             'cols_equal_c_times_ns': bool(torch.equal(r2b[WI][:, NV:], ns * c)),
             'cols_equal_r1b': bool(torch.equal(r2b[WI][:, NV:], r1b[WI][:, NV:])),
             'norm': {'shared': float(sh.norm()), 'ns': float(ns.norm()), 'r2b': float(r2b[WI][:, NV:].norm())}}
        r['norm_matches'] = abs(r['norm']['r2b'] / r['norm']['shared'] - 1) < 1e-5
        m = load_like_training(f'v15_s2p2h_noexp_v15i_r2b{suf(s)}', 'r2b', s)
        st, rs = m.state_dict(), E.load(f'v15_s2p2h_noexp_v15i{suf(s)}', 0, torch.device('cpu'), seed=s).state_dict()
        rest = [k for k in st if k not in r2b]
        r['loaded_equals_ckpt'] = all(torch.equal(st[k], r2b[k]) for k in r2b)
        r['rest_is_G_equal_shared_ep0'] = all(k.startswith('motion_generator_module.') for k in rest) and \
            all(torch.equal(st[k], rs[k]) for k in rest)
        r['passed'] = all(v for v in r.values() if isinstance(v, bool))
        ok &= r['passed']
        R[s] = r
        print(s, r, flush=True)
    R['all_passed'] = ok
    json.dump(R, open(os.path.join(E.OUT, 'make_r2b_v15.json'), 'w'), indent=1)
    print('ALL_PASSED' if ok else 'FAILED', flush=True)


if __name__ == '__main__':
    main()
