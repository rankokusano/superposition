"""
v12 E1/E2 (docs/v12_experiment_log.md Sec.53, post-hoc check for the discussion): encoder-side measures of one
checkpoint, CPU only, same definitions as before:
  * self/other code alignment cos_ov(A1)_sv(A2) (eval_trial_v12.alignment, Sec.8);
  * layout (original / swapped) read linearly from Encoder-1 / Encoder-2 of A-1's view, and A-2's green direction
    from Encoder-2 (enc_colour_v12, Sec.19), on the common test and on v11_s2t_green.
--ckpt_exp/--epoch name the checkpoint data/result/<exp>/0/model/<epoch>.pth. Writes data/result/v12/e12/<exp>_ep<e>.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v12 import alignment  # noqa
from enc_colour_v12 import frames, acc_cv, RES  # noqa
from fig5_extra_v11 import make_ae, enc, cv_r2  # noqa
from eval_m1_v11 import refs  # noqa
from sklearn.preprocessing import PolynomialFeatures  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', required=True)
    ap.add_argument('--ckpt_path', default=None, help='reproduction check: an explicit checkpoint (e.g. v4)')
    a = ap.parse_args()
    torch.set_num_threads(8)
    out = os.path.join(RES, 'v12', 'e12'); os.makedirs(out, exist_ok=True)
    D = {k: frames(k) for k in ('v11_s1_random', 'v11_s2t_green')}
    sv, lay, op, grp = D['v11_s2t_green']
    u = np.stack([refs(np.array([l]), 'Green')[0, 0] for l in lay]) - op
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    poly = PolynomialFeatures(2).fit_transform(op)
    for e in a.epochs:
        ck = a.ckpt_path or os.path.join(RES, a.exp, '0', 'model', f'{e:05d}.pth')
        m, _ = make_ae(ck)
        r = {'exp': a.exp, 'epoch': e, 'alignment': alignment(m)}
        for k, (sv_, lay_, op_, grp_) in D.items():
            S, O = enc(m.self_vision_encoder_module, sv_), enc(m.other_vision_encoder_module, sv_)
            r[k] = {'layout_from_sv_enc': acc_cv(S, lay_, grp_), 'layout_from_ov_enc': acc_cv(O, lay_, grp_)}
            if k == 'v11_s2t_green':
                r[k]['green_dir_from_ov_enc'] = cv_r2(O, u, grp_)
                r[k]['green_dir_from_ov_enc_plus_A2pos'] = cv_r2(np.concatenate([O, poly], 1), u, grp_)
        json.dump(r, open(os.path.join(out, f'{a.exp}_ep{e}.json'), 'w'), indent=1)
        print(a.exp, e, json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
