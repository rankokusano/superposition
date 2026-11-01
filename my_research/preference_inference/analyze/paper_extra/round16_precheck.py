"""
Paper round 16 Sec.1 (docs/paper_instructions_round16.md): checks before any computation. CPU only, no training.
The loss path is stageC_eval's (gradients(): self-vision L1 + feature prediction from h1 + from h2, per step, Q2 given
through p2_override = [0, 0, Q2_t, 1], G still running inside). On the first n episodes of v12_s2_pref_mixed_s0 ep200:
  1 stage-B set (vdirb): this code's Q2 and d(total loss)/d(Q2) at G's point vs the saved StageC/eval arrays;
  2 v11_s2_pref test: per-episode FP (process-2 term) and self-vision L1, mean over t = 0..99, with G's Q2 and with
    Q2 = 0, vs the saved T3 per-episode values (v12/stage2/t1t3/pref_mixed_s0_ep200_per.npz);
  3 substitution: losses with p2_override = G's own Q2 vs no override (must be identical: only Q2 is replaced).
-> data/result/paper_extra/Round16/precheck/summary.json
"""

import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stageC_eval as C  # noqa  (protocol, Hook, p2_input, batches, gradients; registers 'vdirb')

DEV = torch.device('cpu')
OUT = os.path.join(C.RES, 'paper_extra', 'Round16', 'precheck')
EXP, S, EP, N = 'v12_s2_pref_mixed_s0', 0, 200, 50


@torch.no_grad()
def losses(model, g, n, Qvals=None, B=50):
    """per episode and step t = 0..99: (L1 self-vision, FP from h1, FP from h2); Q2 from Qvals or from G."""
    C.util.seed_all(0)
    out = []
    for b0, b1, x, m in C.batches(g, n, B):
        model.init_state(b1 - b0)
        seq = []
        for t in range(100):
            pm = 0 if t == 0 else 1
            p2 = None if Qvals is None else C.p2_input(torch.from_numpy(Qvals[b0:b1, t]))
            o = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=p2)
            tgt = x[:, t + 1]
            st = model.get_state()
            l1 = F.l1_loss(o['self_vision'], tgt, reduction='none').mean((1, 2, 3))
            f1 = F.mse_loss(model.predict_feature(st['self'][0]), model.self_vision_encoder_module(tgt), reduction='none').mean(1)
            f2 = F.mse_loss(model.predict_feature(st['other'][0]), model.other_vision_encoder_module(tgt), reduction='none').mean(1)
            seq.append(torch.stack([l1, f1, f2], 1).numpy())
        out.append(np.stack(seq, 1))
    return np.concatenate(out)                                   # (n, 100, 3)


def main():
    torch.set_num_threads(int(os.environ.get('NT', '8')))
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    R = {'model': f'{EXP} ep{EP}', 'n_episodes': N, 'device': 'cpu'}
    model, _ = C.load_v12(EXP, S, EP, DEV)
    model.eval()
    hook = C.Hook(model.other_vision_encoder_module)
    sv = np.load(os.path.join(C.OUT, 'eval', f'{EXP}_ep{EP}.npz'))
    # 1 stage-B set
    d = C.E.load_set('vdirb')
    g = h5py.File(d['path'], 'r')[d['split']]
    Q = C.forward_states(model, hook, g, N, DEV, zero=False)['Q']
    gt, _ = C.gradients(model, g, N, DEV, Q.astype(np.float32))
    R['1_Q2_maxabs'] = float(np.abs(Q - sv['Q2'][:N]).max())
    R['1_grad_tot_G_maxabs'] = float(np.abs(gt - sv['grad_tot_G'][:N]).max())
    R['1_grad_tot_G_scale'] = float(np.abs(sv['grad_tot_G'][:N]).max())
    R['1_gproj_tot_G_maxabs'] = float(np.abs(-(gt @ C.CPU_) - sv['gproj_tot_G'][:N]).max())
    L_vd = losses(model, g, N, Q.astype(np.float32))
    R['1_loss_vdirb_mean_t1_50'] = {k: float(L_vd[:, 1:51, i].mean()) for i, k in enumerate(('L1', 'FP1', 'FP2'))}
    R['3_sub_vs_none_maxabs'] = float(np.abs(L_vd - losses(model, g, N, None)).max())
    # 2 T3 per-episode values on v11_s2_pref test
    d2 = C.E.load_set('pref')
    g2 = h5py.File(d2['path'], 'r')[d2['split']]
    t3 = np.load(os.path.join(C.RES, 'v12', 'stage2', 't1t3', f'pref_mixed_s{S}_ep{EP}_per.npz'))
    Qp = C.forward_states(model, hook, g2, N, DEV, zero=False)['Q'].astype(np.float32)
    Lg, Lz = losses(model, g2, N, Qp), losses(model, g2, N, np.zeros_like(Qp))
    for nm, L in (('gen', Lg), ('zero', Lz)):
        R[f'2_{nm}_fp_maxabs'] = float(np.abs(L[:, :, 2].mean(1) - t3[f'pref_{nm}_fp'][:N]).max())
        R[f'2_{nm}_self_maxabs'] = float(np.abs(L[:, :, 0].mean(1) - t3[f'pref_{nm}_self'][:N]).max())
    R['2_D_G_fp_t0_99_this'] = float((Lz[:, :, 2] - Lg[:, :, 2]).mean())
    R['2_D_G_fp_t0_99_T3'] = float((t3['pref_zero_fp'][:N] - t3['pref_gen_fp'][:N]).mean())
    R['t_s'] = time.time() - t0
    json.dump(R, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1), flush=True)


if __name__ == '__main__':
    main()
