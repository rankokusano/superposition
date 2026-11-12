"""
Paper round 33 (case A; paper/check/paper_instructions_round33.md; docs/paper_extra_log.md Sec.57; exploratory, outside
the pre-registration; procedure fixed before computing). Compare three ways of making the other's pleasure (A-2's
red - green per range) from the observer's own experience only:
  case 1  internal-state sharing: read-out R (h1 -> own pleasure) learnt on own states, applied to h2;
          (i) process-1 run with the pleasure input set to 0, target = current own pleasure;
          (ii) normal run, target = own pleasure of the NEXT frame; linear (ridge) and a small net (128-64-8).
  case 2  position map M ((own x, y, colour layout read from the own view) -> own pleasure, 3-64-64-8), applied to
          A-2's true position / the position found in A-1's view (round-30 Sec.3 read_a2) with the layout read there.
  case 3  geometry (round-30 Sec.3): A-2's view rendered at the found position / layout, counted with pleasure_np.
  diag    ridge h2 -> A-2's red - green learnt on half of the (b) episodes (answers used; diagnostic only).
Predictors are never changed. Only R, M and the diagnostic ridge are learnt.

Evaluation data
  (b)  v11_s2_pref test (300 episodes x 101 frames, no swap; round 30 Sec.2 (b) / Sec.3 (b)), all frames shown to both
       processes, process-1 motion visible, truth = red - green of A-2's view at frames 1..100.
  (b2) paper_r21_swap data_noswap (600 x 41, A-2 static), same flow, frames 1..40 (descriptive).
  (c)  paper_r21_swap data_swap: round-21 / round-30 swap flow (frames 0..22 shown, swap at 20, 15 frozen steps);
       truth = A-2's view after the swap ((i) frames 23..37, (ii) and cases 2 / 3 frames 24..38); also the no-swap
       counterpart (data_noswap, same frames) to see whether the old direction is followed.

Modes
  check (GPU1)                  reproduction / fact checks -> Round33/check/check.json
  flows --cond c (GPU)          case 1, diag, predicted-image geometry inputs, pairs 0-5 -> Round33/flows/<c>_s<s>.npz
  real (CPU)                    geometry inputs found in the real A-1 views of (b), (b2), (c) -> Round33/real/real.npz
  pos --cond c (CPU)            case 2 (M seeds 0-5) -> Round33/pos/<c>.npz   (needs real and flows)
  geo --part i --n_parts n (CPU, xvfb)  case 3 renders -> Round33/geo/part<i>of<n>.npz   (needs real and flows)
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _PI)
os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')

from v11core.pleasure import pleasure_np  # noqa

DATA = os.path.join(_PI, 'data', 'data')
RES = os.path.join(_PI, 'data', 'result')
OUT = os.path.join(RES, 'paper_extra', 'Round33')
PAIRS = tuple(range(6))
CONDS = ('p50', 'none', 'sr50', 'p80')
EXP = {'p50': 'v12_s1_mixed_l1_s{s}', 'none': 'v12_s1_random_l1_s{s}',
       'sr50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s{s}', 'p80': 'v13_s1_mixed_l1_s{s}'}
S1DATA = {'p50': 'v11_s1_mixed', 'sr50': 'v11_s1_mixed', 'none': 'v11_s1_random', 'p80': 'v13_s1_mixed80'}
N_TRAIN = 1000               # stage-I train episodes 0..999 for R and M
T0, K, TS = 23, 15, 20


def tenv():
    import torch
    import round21_eval as R21
    import round29 as R29m
    import round30 as R30m
    import util
    from p2test_v11 import read_a2
    return torch, R21, R29m, R30m, util, read_a2


def f16(a):
    return np.asarray(a, np.float32).astype(np.float16)


def load_model(cond, s, dev):
    torch, R21, R29m, R30m, util, _ = tenv()
    m, _ = R21.T.load_v12(EXP[cond].format(s=s), s, 200, dev)
    return m.eval()


# =============================================================== flows
def show_states(model, sv, sm, variant, dev, batch=50, want_pred=False):
    """all frames shown to both processes (mask 0), process-1 motion visible (motion_masking off), process-2 input 0;
    variant 'i': p1_override = 0 at every step; 'ii': the model's own pleasure input.
    -> h1, h2 (N, T, 128) float32 after each step t = 0..T-1; pred (N, T-1, 16, 64, 3) uint8 (step t predicts t + 1)."""
    torch, R21, R29m, R30m, util, _ = tenv()
    model.motion_masking = False
    H1, H2, PR = [], [], []
    with torch.no_grad():
        util.seed_all(0)
        for b0 in range(0, sv.shape[0], batch):
            x = R21.scaled(sv[b0:b0 + batch], dev)
            m = torch.from_numpy(sm[b0:b0 + batch]).float().to(dev)
            B, T = x.shape[:2]
            model.init_state(B)
            z = torch.zeros(B, 8, device=dev)
            h1, h2, pr = [], [], []
            for t in range(T):
                o = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0, 0,
                          p1_override=z if variant == 'i' else None)['self_vision']
                st = model.get_state()
                h1.append(st['self'][0].cpu().numpy())
                h2.append(st['other'][0].cpu().numpy())
                if want_pred and t < T - 1:
                    pr.append(np.round(((o + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8))
            H1.append(np.stack(h1, 1))
            H2.append(np.stack(h2, 1))
            if want_pred:
                PR.append(np.stack(pr, 1))
    return np.concatenate(H1), np.concatenate(H2), (np.concatenate(PR) if want_pred else None)


def swap_states(model, D, variant, dev, batch=50):
    """round30.swap_flow (= round21 freeze_flow self / hold, V1) with the states: frames 0..T0-1 shown, then K steps
    with vision masked, motion hidden, p1_override = pleasure of frame T0-1 ('ii') or 0 ('i', also before freezing).
    -> h1, h2 (N, K, 128) at the frozen steps, pred (N, K, 16, 64, 3) float32."""
    torch, R21, R29m, R30m, util, _ = tenv()
    N = D['self_vision'].shape[0]
    H1, H2, PR = [], [], []
    with torch.no_grad():
        for b0 in range(0, N, batch):
            x = R21.scaled(D['self_vision'][b0:b0 + batch], dev)
            m = torch.from_numpy(D['self_motion'][b0:b0 + batch]).float().to(dev)
            B = x.shape[0]
            z = torch.zeros(B, 8, device=dev)
            util.seed_all(0)
            model.init_state(B)
            for t in range(T0):
                model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0, 0, p1_override=z if variant == 'i' else None)
            P1 = z if variant == 'i' else model.compute_pleasure(x[:, T0 - 1])
            hid = torch.ones(B, dtype=torch.bool, device=dev)
            h1, h2, pr = [], [], []
            for k in range(K):
                o = model({'self_vision': x[:, T0 + k], 'self_motion': m[:, T0 + k]}, 1.0, 1.0, p1_override=P1,
                          p2_override=None, hidden_override=hid)['self_vision']
                st = model.get_state()
                h1.append(st['self'][0].cpu().numpy())
                h2.append(st['other'][0].cpu().numpy())
                pr.append(((o + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
            H1.append(np.stack(h1, 1))
            H2.append(np.stack(h2, 1))
            PR.append(np.stack(pr, 1))
    return np.concatenate(H1), np.concatenate(H2), np.concatenate(PR)


def fit_readouts(X, Y, seed, dev):
    """X (n, 128), Y (n, 8) -> {'lin': (W, b), 'mlp': net, 'mu', 'sd'}."""
    torch, R21, R29m, R30m, util, _ = tenv()
    W, b = R29m.ridge_fit(X.astype(np.float64), Y.astype(np.float64))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xg = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(dev)
    Yg = torch.from_numpy(Y.astype(np.float32)).to(dev)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(20):
        perm = torch.randperm(len(Xg), generator=g).to(dev)
        for b0 in range(0, len(Xg), 1024):
            ib = perm[b0:b0 + 1024]
            loss = torch.nn.functional.mse_loss(net(Xg[ib]), Yg[ib])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return {'lin': (W, b), 'mlp': net, 'mu': mu, 'sd': sd, 'mlp_last_loss': float(loss)}


def apply_ro(ro, kind, H):
    """H (..., 128) -> (..., 8) float32."""
    torch = tenv()[0]
    sh = H.shape[:-1]
    X = H.reshape(-1, 128)
    if kind == 'lin':
        W, b = ro['lin']
        return (X.astype(np.float64) @ W + b).astype(np.float32).reshape(sh + (8,))
    dev = next(ro['mlp'].parameters()).device
    with torch.no_grad():
        y = ro['mlp'](torch.from_numpy(((X - ro['mu']) / ro['sd']).astype(np.float32)).to(dev)).cpu().numpy()
    return y.reshape(sh + (8,))


def pairs_xy(H, P, variant):
    """(N, T, 128), (N, T, 8) own pleasure of frames 0..T-1 -> training pairs: 'i' (h_t, P_t), 'ii' (h_t, P_{t+1})."""
    if variant == 'i':
        return H.reshape(-1, 128), P.reshape(-1, 8)
    return H[:, :-1].reshape(-1, 128), P[:, 1:].reshape(-1, 8)


def flows_main(cond):
    torch, R21, R29m, R30m, util, read_a2 = tenv()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    od = os.path.join(OUT, 'flows')
    os.makedirs(od, exist_ok=True)
    ab = R21.calib_common()
    g1 = h5py.File(os.path.join(DATA, S1DATA[cond], 'data.h5'), 'r')
    S1tr = {k: g1['train'][k][:N_TRAIN] for k in ('self_vision', 'self_motion')}
    S1te = {k: g1['test'][k][()] for k in ('self_vision', 'self_motion')}
    P1tr, P1te = R30m.pleasure_all(S1tr['self_vision']), R30m.pleasure_all(S1te['self_vision'])
    te = R30m.load_split('v11_s2_pref', 'test', ('self_vision', 'self_motion', 'self_position'))
    Dns, Dsw = R21.load_data('noswap'), R21.load_data('swap')
    for s in PAIRS:
        t0 = time.time()
        sav = {}
        for v in ('i', 'ii'):
            m = load_model(cond, s, dev)
            H1, _, _ = show_states(m, S1tr['self_vision'], S1tr['self_motion'], v, dev)
            X, Y = pairs_xy(H1, P1tr, v)
            ro = fit_readouts(X, Y, s, dev)
            sav[f'{v}|mlp_last_loss'] = np.array(ro['mlp_last_loss'])
            del H1, X, Y
            # own check (stage-I test of the condition's data)
            H1, _, _ = show_states(load_model(cond, s, dev), S1te['self_vision'], S1te['self_motion'], v, dev)
            for kd in ('lin', 'mlp'):
                sav[f'{v}|{kd}|self'] = f16(apply_ro(ro, kd, H1 if v == 'i' else H1[:, :-1]))
            # (b) v11_s2_pref test, steps 0..100
            H1, H2, PR = show_states(load_model(cond, s, dev), te['self_vision'], te['self_motion'], v, dev, want_pred=(v == 'ii'))
            sel = slice(1, 101) if v == 'i' else slice(0, 100)            # truth frames 1..100
            for kd in ('lin', 'mlp'):
                sav[f'{v}|{kd}|b'] = f16(apply_ro(ro, kd, H2[:, sel]))
                sav[f'{v}|{kd}|b_own'] = f16(apply_ro(ro, kd, H1[:, sel]))
            if v == 'ii':
                sav['geo_b_est'], sav['geo_b_lay'] = R30m.geom_inputs(PR.astype(np.float32) / 255.0,
                                                                      te['self_position'][:, 1:].astype(np.float64), ab, read_a2)
            Hb2 = H2[:, sel]
            del PR
            # (b2) round-21 no-swap data, steps 0..40, truth frames 1..40
            _, H2n, _ = show_states(load_model(cond, s, dev), Dns['self_vision'], Dns['self_motion'], v, dev)
            sel2 = slice(1, 41) if v == 'i' else slice(0, 40)
            for kd in ('lin', 'mlp'):
                sav[f'{v}|{kd}|b2'] = f16(apply_ro(ro, kd, H2n[:, sel2]))
            # (c) swap flow, frozen steps
            m = load_model(cond, s, dev)
            m.motion_masking = True
            _, H2c, PRc = swap_states(m, Dsw, v, dev)
            for kd in ('lin', 'mlp'):
                sav[f'{v}|{kd}|c'] = f16(apply_ro(ro, kd, H2c))
            if v == 'ii':
                sav['geo_c_est'], sav['geo_c_lay'] = R30m.geom_inputs(PRc, Dsw['self_position'][:, T0:T0 + K].astype(np.float64), ab, read_a2)
            # diag: ridge h2 -> A-2's red - green, (b) episodes 0..149 -> 150..299 (answers used)
            Pa2 = R30m.pleasure_all(h5py.File(os.path.join(DATA, 'v11_s2_pref', 'data.h5'), 'r')['test']['other_vision'][:, 1:101])
            W, b = R29m.ridge_fit(Hb2[:150].reshape(-1, 128).astype(np.float64), Pa2[:150].reshape(-1, 8).astype(np.float64))
            sav[f'{v}|diag|b_test'] = f16(Hb2[150:].astype(np.float64) @ W + b)
            sav[f'{v}|diag|c'] = f16(H2c.astype(np.float64) @ W + b)
            del Hb2, H2c, PRc, H2n
        np.savez_compressed(os.path.join(od, f'{cond}_s{s}.npz'), **sav)
        torch.cuda.empty_cache()
        print(f'flows:{cond}:{s} {time.time() - t0:.0f}s {time.strftime("%H:%M")}', flush=True)


# =============================================================== real-image geometry inputs and truths
def real_main():
    torch, R21, R29m, R30m, util, read_a2 = tenv()
    od = os.path.join(OUT, 'real')
    os.makedirs(od, exist_ok=True)
    ab = R21.calib_common()
    g = h5py.File(os.path.join(DATA, 'v11_s2_pref', 'data.h5'), 'r')['test']
    Dns, Dsw = R21.load_data('noswap'), R21.load_data('swap')
    sav = {}
    sv, sp = g['self_vision'][:, 1:101], g['self_position'][:, 1:101].astype(np.float64)
    sav['b_est'], sav['b_lay'] = R30m.geom_inputs(sv.astype(np.float32) / 255.0, sp, ab, read_a2)
    sv, sp = Dns['self_vision'][:, 1:41], Dns['self_position'][:, 1:41].astype(np.float64)
    sav['b2_est'], sav['b2_lay'] = R30m.geom_inputs(sv.astype(np.float32) / 255.0, sp, ab, read_a2)
    fr = slice(T0 + 1, T0 + K + 1)
    sv, sp = Dsw['self_vision'][:, fr], Dsw['self_position'][:, fr].astype(np.float64)
    sav['c_est'], sav['c_lay'] = R30m.geom_inputs(sv.astype(np.float32) / 255.0, sp, ab, read_a2)
    # truths (A-2's red - green) and true positions / layouts
    sav['truth_b'] = R30m.pleasure_all(g['other_vision'][:, 1:101])
    sav['truth_b2'] = R30m.pleasure_all(Dns['other_vision'][:, 1:41])
    sav['truth_c_i'] = R30m.pleasure_all(Dsw['other_vision'][:, T0:T0 + K])
    sav['truth_c'] = R30m.pleasure_all(Dsw['other_vision'][:, fr])
    sav['truth_c_old_i'] = R30m.pleasure_all(Dns['other_vision'][:, T0:T0 + K])
    sav['truth_c_old'] = R30m.pleasure_all(Dns['other_vision'][:, fr])
    sav['pos_b'], sav['lay_b'] = g['other_position'][:, 1:101], g['layout'][()]
    sav['pos_b2'], sav['lay_b2'] = Dns['other_position'][:, 1:41], Dns['layout']
    sav['pos_c'], sav['lay_c'] = Dsw['other_position'][:, fr], Dsw['layout'] ^ 1   # layout after the swap
    sav['own_pleasure_b'] = R30m.pleasure_all(g['self_vision'][:, 1:101])
    np.savez_compressed(os.path.join(od, 'real.npz'), **sav)
    print('real done', {k: v.shape for k, v in sav.items()}, flush=True)


# =============================================================== case 2 (position map)
def make_M():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))


def M_in(pos, lay):
    """pos (..., 2) world units, lay (...) code 0 / 1 (or -1 = not found) -> (..., 3) float32, NaN where not found."""
    x = np.concatenate([np.asarray(pos, np.float64) / 10.0, (2.0 * np.asarray(lay, np.float64) - 1.0)[..., None]], -1)
    bad = (np.asarray(lay) < 0) | ~np.isfinite(np.asarray(pos, np.float64)).all(-1)
    x[bad] = np.nan
    return x.astype(np.float32)


def M_apply(net, X):
    import torch
    sh = X.shape[:-1]
    Xf = X.reshape(-1, 3)
    ok = np.isfinite(Xf).all(1)
    out = np.full((len(Xf), 8), np.nan, np.float32)
    with torch.no_grad():
        out[ok] = net(torch.from_numpy(Xf[ok])).numpy()
    return out.reshape(sh + (8,))


def pos_main(cond):
    torch, R21, R29m, R30m, util, read_a2 = tenv()
    torch.set_num_threads(4)
    od = os.path.join(OUT, 'pos')
    os.makedirs(od, exist_ok=True)
    g1 = h5py.File(os.path.join(DATA, S1DATA[cond], 'data.h5'), 'r')
    sav = {}
    sets = {}
    for sp_ in ('train', 'test'):
        n = N_TRAIN if sp_ == 'train' else None
        sv = g1[sp_]['self_vision'][:n]
        lay = np.array([[R30m.layout_est(sv[i, t].astype(np.float32) / 255.0) for t in range(sv.shape[1])] for i in range(sv.shape[0])], np.int8)
        sets[sp_] = (M_in(g1[sp_]['self_position'][:n], lay), R30m.pleasure_all(sv), lay, g1[sp_]['layout'][:n])
        sav[f'own_{sp_}_layout_found'] = np.array((lay >= 0).mean())
        sav[f'own_{sp_}_layout_correct'] = np.array((lay[lay >= 0] == np.repeat(sets[sp_][3][:, None], lay.shape[1], 1)[lay >= 0]).mean())
    X, Y = sets['train'][0].reshape(-1, 3), sets['train'][1].reshape(-1, 8)
    ok = np.isfinite(X).all(1)
    X, Y = torch.from_numpy(X[ok]), torch.from_numpy(Y[ok])
    real = np.load(os.path.join(OUT, 'real', 'real.npz'))
    for s in PAIRS:
        t0 = time.time()
        torch.manual_seed(s)
        net = make_M()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(s)
        for ep in range(30):
            perm = torch.randperm(len(X), generator=gen)
            for b0 in range(0, len(X), 1024):
                ib = perm[b0:b0 + 1024]
                loss = torch.nn.functional.mse_loss(net(X[ib]), Y[ib])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        fl = np.load(os.path.join(OUT, 'flows', f'{cond}_s{s}.npz'))
        sav[f's{s}|self'] = f16(M_apply(net, sets['test'][0]))
        sav[f's{s}|b_true'] = f16(M_apply(net, M_in(real['pos_b'], np.repeat(real['lay_b'][:, None], 100, 1))))
        sav[f's{s}|b_found'] = f16(M_apply(net, M_in(real['b_est'], real['b_lay'])))
        sav[f's{s}|b_found_pred'] = f16(M_apply(net, M_in(fl['geo_b_est'], fl['geo_b_lay'])))
        sav[f's{s}|b2_true'] = f16(M_apply(net, M_in(real['pos_b2'], np.repeat(real['lay_b2'][:, None], 40, 1))))
        sav[f's{s}|b2_found'] = f16(M_apply(net, M_in(real['b2_est'], real['b2_lay'])))
        sav[f's{s}|c_true'] = f16(M_apply(net, M_in(real['pos_c'], np.repeat(real['lay_c'][:, None], K, 1))))
        sav[f's{s}|c_found'] = f16(M_apply(net, M_in(fl['geo_c_est'], fl['geo_c_lay'])))
        sav[f's{s}|c_found_real'] = f16(M_apply(net, M_in(real['c_est'], real['c_lay'])))
        sav[f's{s}|last_loss'] = np.array(float(loss))
        print(f'pos:{cond}:{s} {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(os.path.join(od, f'{cond}.npz'), **sav)


# =============================================================== case 3 (geometry renders)
def geo_main(part, n_parts):
    from sim_v10 import creator
    from sim_v10.util import load_config
    from sim_v10 import layout as L
    cfg = load_config(os.path.join(_PI, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml'))
    env = creator.create_environment(cfg.environment)
    env.init()
    env.off_display()

    def render_other(sp, op, lay):
        env.set_layout(L.CODE_LAYOUT[int(lay)])
        env.self_agent.p = np.asarray(sp, dtype=np.float64)
        env.other_agent.p = np.asarray(op, dtype=np.float64)
        env.world.set_camera('other')
        env.world.draw(env.self_agent, env.other_agent)
        return np.round(env.world.capture() * 255).astype(np.uint8)

    import round21_eval as R21
    g = h5py.File(os.path.join(DATA, 'v11_s2_pref', 'data.h5'), 'r')['test']
    Dns, Dsw = R21.load_data('noswap'), R21.load_data('swap')
    sp = {'b': g['self_position'][:, 1:101].astype(np.float64), 'b2': Dns['self_position'][:, 1:41].astype(np.float64),
          'c': Dsw['self_position'][:, T0 + 1:T0 + K + 1].astype(np.float64)}
    real = np.load(os.path.join(OUT, 'real', 'real.npz'))
    jobs = [('real_b', 'b', None), ('real_b2', 'b2', None), ('real_c', 'c', None)]
    jobs += [(f'pred_c_{c}_s{s}', 'c', (c, s)) for c in CONDS for s in PAIRS]
    jobs += [(f'pred_b_p50_s{s}', 'b', ('p50', s)) for s in PAIRS]
    mine = [j for i, j in enumerate(jobs) if i % n_parts == part]
    od = os.path.join(OUT, 'geo')
    os.makedirs(od, exist_ok=True)
    sav = {}
    if part == 0:   # re-render check: A-2's view at its true position / layout equals the stored view
        bad = sum(int(not np.array_equal(render_other(sp['b'][n, t], g['other_position'][n, t + 1], g['layout'][n]),
                                         g['other_vision'][n, t + 1])) for n in range(20) for t in (0, 50, 99))
        sav['rerender_mismatch_b'] = np.array(bad)
        print('rerender mismatches', bad, flush=True)
    for name, sc, src in mine:
        t0 = time.time()
        if src is None:
            est, lay = real[f'{sc}_est'], real[f'{sc}_lay']
        else:
            fl = np.load(os.path.join(OUT, 'flows', f'{src[0]}_s{src[1]}.npz'))
            est, lay = fl[f'geo_{sc}_est'], fl[f'geo_{sc}_lay']
        N, T = est.shape[:2]
        P = np.full((N, T, 8), np.nan, np.float32)
        for n in range(N):
            for t in range(T):
                if np.all(np.isfinite(est[n, t])) and lay[n, t] >= 0:
                    P[n, t] = pleasure_np(render_other(sp[sc][n, t], np.clip(est[n, t], -10, 10), lay[n, t]))
        sav[name] = f16(P)
        print(f'geo:{name} {time.time() - t0:.0f}s {time.strftime("%H:%M")}', flush=True)
    np.savez_compressed(os.path.join(od, f'part{part}of{n_parts}.npz'), **sav)


# =============================================================== check
def check():
    torch, R21, R29m, R30m, util, read_a2 = tenv()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    os.makedirs(os.path.join(OUT, 'check'), exist_ok=True)
    out = {}
    D = R21.load_data('swap', 50)
    m = load_model('p50', 0, dev)
    _, ref = R30m.swap_flow(m, D, dev)
    m = load_model('p50', 0, dev)
    m.motion_masking = True
    _, _, pr = swap_states(m, D, 'ii', dev)
    out['swap_states_ii_vs_round30_swap_flow_max_abs'] = float(np.abs(ref - pr).max())
    te = R30m.load_split('v11_s2_pref', 'test', ('self_vision', 'self_motion'))
    sv, sm = te['self_vision'][:50], te['self_motion'][:50]
    _, ref = R30m.show_flow(load_model('p50', 0, dev), sv, sm, dev)
    h1i, h2i, _ = show_states(load_model('p50', 0, dev), sv, sm, 'i', dev)
    h1, h2, pr = show_states(load_model('p50', 0, dev), sv, sm, 'ii', dev, want_pred=True)
    out['show_states_ii_vs_round30_show_flow_max_abs'] = float(np.abs(ref.astype(np.int16) - pr.astype(np.int16)).max())
    out['h2_same_in_i_and_ii_max_abs'] = float(np.abs(h2i - h2).max())
    out['h1_differs_i_vs_ii_mean_abs'] = float(np.abs(h1i - h1).mean())
    for c in CONDS:
        mm = load_model(c, 0, dev)
        out[f'{c}|class'] = type(mm).__name__
        out[f'{c}|pleasure_scale'] = float(getattr(mm, 'pleasure_scale', float('nan')))
    json.dump(out, open(os.path.join(OUT, 'check', 'check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('check', 'flows', 'real', 'pos', 'geo'))
    ap.add_argument('--cond', default='p50')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--n_parts', type=int, default=1)
    a = ap.parse_args()
    if a.mode == 'flows':
        flows_main(a.cond)
    elif a.mode == 'pos':
        pos_main(a.cond)
    elif a.mode == 'geo':
        geo_main(a.part, a.n_parts)
    else:
        {'check': check, 'real': real_main}[a.mode]()
