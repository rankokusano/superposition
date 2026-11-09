"""
Paper round 31 (paper/check/paper_instructions_round31.md; docs/paper_extra_log.md Sec.55; exploratory, outside the
pre-registration; procedure fixed before computing). Re-check the decoding of the other's view in the same form as
the original paper (Noguchi et al. 2022, Fig.5d/5e): A-1's view -> Encoder-2 -> existing Visual Decoder (v12_fig5_s0,
learnt Encoder-1(A-1 view) -> A-1 view) -> image v2hat, compared with A-2's real view, on the grid placements
(A-1 and A-2 at every integer position in [-9, 9]^2, overlapping placements excluded, both colour layouts).
No training except the Sec.2.4 diagnostic ridge read-outs; predictor weights are never changed.

Modes
  check (GPU1, xvfb) encoders of the 24 stage-I predictors (p50 / none / sr50 / p80 x pairs 0-5) and of the
        decoder's Autoencoder identical; grid indexing; float grid == uint8 / 255; re-render of grid views with sim_v10;
        -> Round31/check/check.json
  main  (GPU) all grid placements (both layouts, margin 1, including the 361 overlapping ones flagged) -> per placement:
        Encoder-1 / Encoder-2 features of A-1's view, v1hat = D(E1), v2hat = D(E2); pixel differences; per-range
        measures of v1, v2, v1hat, v2hat; unclamped differences for the reproduction check of the v12 Fig.5 json
        -> Round31/main/grid.npz, examples.npz
  read  (CPU) Sec.2.4 ridge read-outs (alpha 1) Encoder-2 (and Encoder-1, descriptive) -> A-2's per-range red - green
        and coloured-box amount, learnt on a random half of the placements (seed 0), evaluated on the other half
        -> Round31/read/read.npz
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

from v11core.pleasure import pleasure_np, color_mask  # noqa

DATA = os.path.join(_PI, 'data', 'data')
RES = os.path.join(_PI, 'data', 'result')
OUT = os.path.join(RES, 'paper_extra', 'Round31')
GRIDS = {'original': os.path.join(DATA, 'grid', 'data.h5'), 'swap_rg': os.path.join(DATA, 'v10_swap_grid', 'data.h5')}
SIZE = 21
KEEP = np.arange(1, SIZE - 1)                                   # positions -9..9 (margin 1, as the original code)
KEEP2 = (KEEP[:, None] * SIZE + KEEP[None, :]).reshape(-1)      # 361 grid indices
PAIRS = tuple(range(6))
EXP = {'p50': 'v12_s1_mixed_l1_s{s}', 'none': 'v12_s1_random_l1_s{s}',
       'sr50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s{s}', 'p80': 'v13_s1_mixed_l1_s{s}'}
DEC_EXISTING = ('v12_fig5_s0', 0, 100)
V12_JSON = os.path.join(RES, 'v12', 'main', 'fig5', 'fig5_v12_fig5_s0_s0_ep100.json')
EX_CONFIGS = [((-5, 5), (5, -5)), ((5, 5), (-5, -5)), ((0, -7), (0, 7)), ((-7, 0), (7, 0))]   # (A-1, A-2), as fig5_v11


def gpos(i):
    return np.stack(np.divmod(np.asarray(i), SIZE), -1) - 10


def u8(a):
    return a if a.dtype == np.uint8 else np.round(a * 255).astype(np.uint8)


def measures(img):
    """img float [0,1] (N, 16, 64, 3) -> dict of (N, 8): per range (8 columns x 16 rows)
    col_thr  : (#Red - #Green) / 128 with the self pleasure thresholds (pleasure_np on the uint8 image)
    pos_thr  : (#Red + #Green) / 128, same thresholds (colour-agnostic coloured-box amount)
    col_cont : mean of (R - G) * (1 - B)          (red +1, green -1, olive / grey / blue / cyan / black / white 0)
    pos_cont : mean of max(0, max(R, G) - B)      (red, green 1, olive 0.5, grey / blue / cyan / black / white 0)
    (channel 0 = 'Red', the project convention)."""
    u = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
    rg = (color_mask(u, 'Red').astype(np.float32) + color_mask(u, 'Green')).sum(-2)          # (N, 64)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rng = lambda x: x.reshape(x.shape[0], 16, 8, 8).mean((1, 3))  # noqa  (N, 16, 64) -> (N, 8)
    return {'col_thr': pleasure_np(u), 'pos_thr': (rg.reshape(-1, 8, 8).sum(-1) / 128.0).astype(np.float32),
            'col_cont': rng((R - G) * (1 - B)).astype(np.float32),
            'pos_cont': rng(np.maximum(0, np.maximum(R, G) - B)).astype(np.float32)}


def tenv():
    import torch
    import round21_eval as R21
    import round29 as R29m
    import round30 as R30m
    return torch, R21, R29m, R30m


# =============================================================== main
def main_run():
    torch, R21, R29m, R30m = tenv()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    od = os.path.join(OUT, 'main')
    os.makedirs(od, exist_ok=True)
    m0, _ = R21.T.load_v12(EXP['p50'].format(s=0), 0, 200, dev)
    m0.eval()
    dec = R30m.existing_decoder(dev).ae_vision_decoder_module

    def dec_raw(F):
        flat = torch.from_numpy(np.ascontiguousarray(F)).float().to(dev)
        with torch.no_grad():
            y = dec(flat)
        return ((y + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()

    D = {k: [] for k in ('layout', 'o', 's', 'overlap', 'F1', 'F2')}
    DK = ('h2_v2', 'h2_v1', 'h1_v1', 'h1_v2', 'h2_v2n', 'h2_v1n', 'h1_v1n', 'h1_v2n', 'raw_h2_v2n', 'raw_h1_v2n',
          'raw_h1_v1n', 'raw_h2_v1n')
    for k in DK:
        D['d_' + k] = []
    M = {}
    ex = {}
    for li, (lay, path) in enumerate(GRIDS.items()):
        g = h5py.File(path, 'r')['train']
        t0 = time.time()
        for o in KEEP2:
            sv, ov = u8(g['self_vision'][o][KEEP2]), u8(g['other_vision'][o][KEEP2])
            svn, ovn = u8(g['self_vision_no_agent'][o][KEEP2]), u8(g['other_vision_no_agent'][o][KEEP2])
            F1, F2 = R29m.enc1(m0, sv, dev), R29m.enc2(m0, sv, dev)
            r1, r2 = dec_raw(F1), dec_raw(F2)
            h1, h2 = np.clip(r1, 0, 1), np.clip(r2, 0, 1)
            f = lambda a: a.astype(np.float32) / 255.0  # noqa
            v1, v2, v1n, v2n = f(sv), f(ov), f(svn), f(ovn)
            dd = lambda a, b: np.abs(a - b).mean((1, 2, 3))  # noqa
            vals = {'h2_v2': dd(h2, v2), 'h2_v1': dd(h2, v1), 'h1_v1': dd(h1, v1), 'h1_v2': dd(h1, v2),
                    'h2_v2n': dd(h2, v2n), 'h2_v1n': dd(h2, v1n), 'h1_v1n': dd(h1, v1n), 'h1_v2n': dd(h1, v2n),
                    'raw_h2_v2n': dd(r2, v2n), 'raw_h1_v2n': dd(r1, v2n), 'raw_h1_v1n': dd(r1, v1n), 'raw_h2_v1n': dd(r2, v1n)}
            for k in DK:
                D['d_' + k].append(vals[k].astype(np.float32))
            for nm, img in (('v1', v1), ('v2', v2), ('h1', h1), ('h2', h2)):
                for mk, mv in measures(img).items():
                    M.setdefault(f'{nm}_{mk}', []).append(mv)
            D['layout'].append(np.full(len(KEEP2), li, np.int8))
            D['o'].append(np.full(len(KEEP2), o, np.int16))
            D['s'].append(KEEP2.astype(np.int16))
            D['overlap'].append(KEEP2 == o)
            D['F1'].append(F1.astype(np.float32))
            D['F2'].append(F2.astype(np.float32))
            for a1, a2 in EX_CONFIGS + [(a2_, a1_) for a1_, a2_ in EX_CONFIGS]:
                oi = (a2[0] + 10) * SIZE + (a2[1] + 10)
                if oi == o:
                    j = int(np.where(KEEP2 == (a1[0] + 10) * SIZE + (a1[1] + 10))[0][0])
                    ex[f'{lay}|A1{a1}|A2{a2}'] = np.stack([sv[j], ov[j], u8(h1[j]), u8(h2[j])])
        print(f'main:{lay} {time.time() - t0:.0f}s {time.strftime("%H:%M")}', flush=True)
    out = {k: np.concatenate(v) for k, v in D.items()}
    out.update({k: np.concatenate(v) for k, v in M.items()})
    np.savez(os.path.join(od, 'grid.npz'), **out)
    np.savez_compressed(os.path.join(od, 'examples.npz'), **ex)
    print('main done', {k: v.shape for k, v in out.items() if k in ('F2', 'd_h2_v2', 'v2_col_thr')}, flush=True)


# =============================================================== Sec.2.4 read-outs
def read_run():
    sys.path.insert(0, _HERE)
    import round29 as R29m
    od = os.path.join(OUT, 'read')
    os.makedirs(od, exist_ok=True)
    z = np.load(os.path.join(OUT, 'main', 'grid.npz'))
    ok = ~z['overlap']
    idx = np.where(ok)[0]
    rs = np.random.RandomState(0)
    perm = rs.permutation(idx)
    tr, te = np.sort(perm[:len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    sav = {'train_idx': tr, 'test_idx': te}
    for fn in ('F2', 'F1'):
        X = z[fn].astype(np.float64)
        for tn in ('v2_col_thr', 'v2_pos_thr', 'v2_col_cont', 'v2_pos_cont'):
            Y = z[tn].astype(np.float64)
            W, b = R29m.ridge_fit(X[tr], Y[tr])
            sav[f'{fn}|{tn}'] = (X[te] @ W + b).astype(np.float32)
    np.savez_compressed(os.path.join(od, 'read.npz'), **sav)
    print('read done', len(tr), len(te), flush=True)


# =============================================================== check
def check():
    torch, R21, R29m, R30m = tenv()
    dev = torch.device('cuda:0')
    os.makedirs(os.path.join(OUT, 'check'), exist_ok=True)
    out = {}
    ae = R30m.existing_decoder(dev).state_dict()
    ks = [k for k in ae if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module'))]
    same = {}
    for c in EXP:
        for s in PAIRS:
            m, _ = R21.T.load_v12(EXP[c].format(s=s), s, 200, dev)
            sd = m.state_dict()
            same[f'{c}_s{s}'] = bool(all(torch.equal(ae[k].cpu(), sd[k].cpu()) for k in ks))
    out['encoder_tensors_compared'] = len(ks)
    out['encoders_identical_to_decoder_autoencoder'] = same
    out['encoders_identical_all_24'] = bool(all(same.values()))
    for lay, path in GRIDS.items():
        g = h5py.File(path, 'r')['train']
        sp, op = g['self_position'][()], g['other_position'][()]
        ii = np.arange(SIZE * SIZE)
        out[f'{lay}|index_ok'] = bool(np.array_equal(sp, np.broadcast_to(gpos(ii)[None], sp.shape)) and
                                      np.array_equal(op, np.broadcast_to(gpos(ii)[:, None], op.shape)))
        v = g['self_vision'][100]
        out[f'{lay}|dtype'] = str(v.dtype)
        if v.dtype != np.uint8:
            out[f'{lay}|float_vs_u8_max_abs'] = float(np.abs(u8(v).astype(np.float32) / 255.0 - v).max())
        P = pleasure_np(u8(g['other_vision'][200]))
        out[f'{lay}|other_view_range_nonzero_frac_o200'] = [float(x) for x in (P != 0).mean(0)]
    # re-render with sim_v10 (A-1 at s, A-2 at o; both cameras; agents drawn)
    from sim_v10 import creator
    from sim_v10.util import load_config
    from sim_v10 import layout as L
    cfg = load_config(os.path.join(_PI, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml'))
    env = creator.create_environment(cfg.environment)
    env.init()
    env.off_display()
    rs = np.random.RandomState(1)
    for lay, path in GRIDS.items():
        g = h5py.File(path, 'r')['train']
        env.set_layout(L.ORIGINAL if lay == 'original' else L.SWAP_RG)
        mis = []
        for _ in range(10):
            o, s = rs.choice(KEEP2, 2, replace=False)
            env.self_agent.p = gpos(s).astype(np.float64)
            env.other_agent.p = gpos(o).astype(np.float64)
            for cam, key in (('self', 'self_vision'), ('other', 'other_vision')):
                env.world.set_camera(cam)
                env.world.draw(env.self_agent, env.other_agent)
                r = np.round(env.world.capture() * 255).astype(np.int16)
                mis.append(int(np.abs(r - u8(g[key][o, s]).astype(np.int16)).max()))
        out[f'{lay}|rerender_max_abs_diff_u8'] = mis
    json.dump(out, open(os.path.join(OUT, 'check', 'check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('check', 'main', 'read'))
    a = ap.parse_args()
    {'check': check, 'main': main_run, 'read': read_run}[a.mode]()
