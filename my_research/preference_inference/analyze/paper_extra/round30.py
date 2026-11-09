"""
Paper round 30 (paper/check/paper_instructions_round30.md; docs/paper_extra_log.md Sec.54; exploratory, outside the
pre-registration; procedure fixed before computing). Can the other's pleasure (red - green per range as seen by A-2)
be made in the image space? Predictors are never changed; the only training is the Sec.1 CNN and the Sec.2 decoder
(descriptive new decoder; the main decoder is the existing paper-recipe Visual Decoder data/result/v12_fig5_s0).

Modes
  s1    (GPU) CNN image -> 8-range red - green, learnt on the p50 stage-I data (v11_s1_mixed train, A-1's own views
        only), seeds 0-5; evaluated on (a) own views and (b) A-2's real views (other_vision) of the test split.
        Hand-made upper bound (pleasure_np on the image) and the round-29 linear read-out (ridge on Encoder-1) alongside.
        -> Round30/s1/cnn_s<s>.pth, s1_s<s>.npz (sufficient statistics), lin.npz, hand.npz
  dec   (GPU) descriptive new decoder: Encoder-2(A-1 view t) -> A-2's view t (L1), per data (mixed, random) x seed 0-5
        -> Round30/dec/<data>_s<s>.pth
  s2 --cond c (GPU) scenes (a) real Encoder-2 features, (b) predicted other-view features in the shown flow on
        v11_s2_pref test, (c) round-21 swap, frames 0..22 shown, 15 frozen steps (round 21 self path, p1 hold, p2 0);
        decoders: existing (main) and new (descriptive); read-outs: CNN of the same seed, hand-made on the decoded image.
        Also the predicted A-1 images' A-2 position / layout estimates for Sec.3 (p50). -> Round30/s2/<c>_s<s>.npz
  s3 --part i --n_parts n (CPU, xvfb) Sec.3 geometry: A-2 position (read_a2, A-1's true position) and layout (side of the
        red / green direction) from A-1's real or predicted image, A-2's view rendered there and counted with
        pleasure_np -> Round30/s3/part<i>of<n>.npz
  check (GPU1) facts and reproduction checks -> Round30/check/check.json
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

from v11core.pleasure import pleasure_np, COL_ANGLES  # noqa
from sim_v10 import layout as L  # noqa

DATA = os.path.join(_PI, 'data', 'data')
RES = os.path.join(_PI, 'data', 'result')
OUT = os.path.join(RES, 'paper_extra', 'Round30')
R29 = os.path.join(RES, 'paper_extra', 'Round29')
PAIRS = tuple(range(6))
CONDS = ('p50', 'none', 'sr50')
EXP = {'p50': 'v12_s1_mixed_l1_s{s}', 'none': 'v12_s1_random_l1_s{s}',
       'sr50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s{s}'}
S1DATA = {'p50': 'v11_s1_mixed', 'sr50': 'v11_s1_mixed', 'none': 'v11_s1_random'}
DEC_EXISTING = ('v12_fig5_s0', 0, 100)
T0, K = 23, 15
BINSETS = {'c0167': [0, 1, 6, 7], 'all': list(range(8)), **{f'b{b}': [b] for b in range(8)}}


def tenv():
    import torch
    import round21_eval as R21   # T = eval_trial_v12 (load_v12), scaled, calib_common, load_data
    import util
    from p2test_v11 import read_a2
    return torch, R21, util, read_a2


def suff(pred, targ, bins):
    """per-episode sufficient statistics over (time, bins) -> (N, 7): n, Sx, Sy, Sxx, Syy, Sxy, SSE (NaN rows dropped)."""
    x = pred[..., bins].reshape(pred.shape[0], -1).astype(np.float64)
    y = targ[..., bins].reshape(targ.shape[0], -1).astype(np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(ok, x, 0), np.where(ok, y, 0)
    return np.stack([ok.sum(1).astype(np.float64), x.sum(1), y.sum(1), (x * x).sum(1), (y * y).sum(1), (x * y).sum(1),
                     ((y - x) ** 2).sum(1)], 1)


def all_suff(pred, targ):
    return {bn: suff(pred, targ, b) for bn, b in BINSETS.items()}


def pleasure_all(v, chunk=200):
    return np.concatenate([pleasure_np(v[c:c + chunk]) for c in range(0, v.shape[0], chunk)]).astype(np.float32)


def load_split(ds, split, keys=('self_vision', 'other_vision', 'self_motion')):
    g = h5py.File(os.path.join(DATA, ds, 'data.h5'), 'r')[split]
    return {k: g[k][()] for k in keys}


# =============================================================== Sec.1 CNN
def make_cnn():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 4 * 16, 8))


def to_x(imgs, dev):
    """uint8 (B, 16, 64, 3) or float [0,1] -> (B, 3, 16, 64) in [-1, 1] (the predictors' scaling)."""
    import torch
    x = torch.as_tensor(imgs, device=dev)
    x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
    return (x * 2 - 1).permute(0, 3, 1, 2)


def cnn_apply(net, imgs, dev, chunk=4096):
    import torch
    flat = imgs.reshape((-1,) + imgs.shape[-3:])
    out = []
    with torch.no_grad():
        for c in range(0, flat.shape[0], chunk):
            out.append(net(to_x(flat[c:c + chunk], dev)).cpu().numpy())
    return np.concatenate(out).reshape(imgs.shape[:-3] + (8,))


def s1_main():
    torch, R21, util, _ = tenv()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    od = os.path.join(OUT, 's1')
    os.makedirs(od, exist_ok=True)
    tr, te = load_split('v11_s1_mixed', 'train'), load_split('v11_s1_mixed', 'test')
    Ptr = pleasure_all(tr['self_vision'])
    Pa1, Pa2 = pleasure_all(te['self_vision']), pleasure_all(te['other_vision'])
    Xtr = tr['self_vision'].reshape(-1, 16, 64, 3)
    Ytr = torch.from_numpy(Ptr.reshape(-1, 8)).to(dev)
    Xg = torch.from_numpy(Xtr).to(dev)
    run = {}
    # hand-made upper bound (the Sec.0-3 computation on the image itself)
    np.savez_compressed(os.path.join(od, 'hand.npz'), **{f'own|{k}': v for k, v in all_suff(pleasure_all(te['self_vision']), Pa1).items()},
                        **{f'other|{k}': v for k, v in all_suff(pleasure_all(te['other_vision']), Pa2).items()})
    # round-29 linear read-out (ridge alpha 1 on Encoder-1), recomputed for all range sets
    m, _ = R21.T.load_v12(EXP['p50'].format(s=0), 0, 200, dev)
    import round29 as R29m
    F = R29m.enc1(m, tr['self_vision'], dev)
    W, b = R29m.ridge_fit(F.reshape(-1, 64).astype(np.float64), Ptr.reshape(-1, 8).astype(np.float64))
    lin_own = R29m.enc1(m, te['self_vision'], dev).astype(np.float64) @ W + b
    lin_oth = R29m.enc1(m, te['other_vision'], dev).astype(np.float64) @ W + b
    np.savez_compressed(os.path.join(od, 'lin.npz'), **{f'own|{k}': v for k, v in all_suff(lin_own, Pa1).items()},
                        **{f'other|{k}': v for k, v in all_suff(lin_oth, Pa2).items()})
    del F
    for s in PAIRS:
        t0 = time.time()
        torch.manual_seed(s)
        net = make_cnn().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        g = torch.Generator(device='cpu').manual_seed(s)
        losses = []
        for ep in range(3):
            perm = torch.randperm(Xg.shape[0], generator=g).to(dev)
            tot = 0.0
            for b0 in range(0, Xg.shape[0], 512):
                ib = perm[b0:b0 + 512]
                loss = torch.nn.functional.mse_loss(net(to_x(Xg[ib], dev)), Ytr[ib])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(ib)
            losses.append(tot / Xg.shape[0])
        net.eval()
        torch.save(net.state_dict(), os.path.join(od, f'cnn_s{s}.pth'))
        own, oth = cnn_apply(net, te['self_vision'], dev), cnn_apply(net, te['other_vision'], dev)
        np.savez_compressed(os.path.join(od, f's1_s{s}.npz'), **{f'own|{k}': v for k, v in all_suff(own, Pa1).items()},
                            **{f'other|{k}': v for k, v in all_suff(oth, Pa2).items()})
        run[f's{s}'] = {'train_mse': losses, 'sec': time.time() - t0}
        print(f's1:{s} {time.time() - t0:.0f}s {losses}', flush=True)
    json.dump(run, open(os.path.join(od, 's1_run.json'), 'w'), indent=1)


def load_cnn(s, dev):
    import torch
    net = make_cnn().to(dev)
    net.load_state_dict(torch.load(os.path.join(OUT, 's1', f'cnn_s{s}.pth'), map_location=dev))
    return net.eval()


# =============================================================== decoders
def existing_decoder(dev):
    import torch
    from model.model import Autoencoder
    from config_util import load_config
    cfg = load_config(os.path.join(_PI, 'config', 'exp', DEC_EXISTING[0] + '.yml'))
    mcfg = load_config(os.path.join(_PI, 'config', 'model', cfg.model.name, cfg.model.config + '.yml'))
    ae = Autoencoder(mcfg).to(dev)
    sd = torch.load(os.path.join(RES, DEC_EXISTING[0], str(DEC_EXISTING[1]), 'model', f'{DEC_EXISTING[2]:05d}.pth'),
                    map_location='cpu')['model']
    ae.load_state_dict(sd)
    return ae.eval()


def new_decoder_module(dev, seed):
    import torch
    torch.manual_seed(seed)
    ae = existing_decoder(dev)                      # same structure; re-initialised below
    dec = ae.ae_vision_decoder_module
    for mod in dec.modules():
        if hasattr(mod, 'reset_parameters') and mod is not dec:
            mod.reset_parameters()
    return dec


def decode(dec, feats, dev, chunk=4096):
    """feats (..., 64) -> images float32 (..., 16, 64, 3) in [0, 1]."""
    import torch
    flat = feats.reshape(-1, feats.shape[-1])
    out = []
    with torch.no_grad():
        for c in range(0, flat.shape[0], chunk):
            y = dec(torch.from_numpy(np.ascontiguousarray(flat[c:c + chunk])).float().to(dev))
            out.append(((y + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out).reshape(feats.shape[:-1] + (16, 64, 3))


def dec_main():
    torch, R21, util, _ = tenv()
    import round29 as R29m
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    od = os.path.join(OUT, 'dec')
    os.makedirs(od, exist_ok=True)
    m, _ = R21.T.load_v12(EXP['p50'].format(s=0), 0, 200, dev)
    run = {}
    for ds in ('v11_s1_mixed', 'v11_s1_random'):
        tr = load_split(ds, 'train', ('self_vision', 'other_vision'))
        F = torch.from_numpy(R29m.enc2(m, tr['self_vision'], dev).reshape(-1, 64)).to(dev)
        Y = torch.from_numpy(tr['other_vision'].reshape(-1, 16, 64, 3)).to(dev)
        for s in PAIRS:
            t0 = time.time()
            dec = new_decoder_module(dev, s).train()
            opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
            g = torch.Generator(device='cpu').manual_seed(s)
            losses = []
            for ep in range(3):
                perm = torch.randperm(F.shape[0], generator=g).to(dev)
                tot = 0.0
                for b0 in range(0, F.shape[0], 256):
                    ib = perm[b0:b0 + 256]
                    loss = torch.abs(dec(F[ib]) - to_x(Y[ib], dev)).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss) * len(ib)
                losses.append(tot / F.shape[0])
            torch.save(dec.state_dict(), os.path.join(od, f'{ds}_s{s}.pth'))
            run[f'{ds}_s{s}'] = {'train_l1': losses, 'sec': time.time() - t0}
            print(f'dec:{ds}:{s} {time.time() - t0:.0f}s {losses}', flush=True)
        del F, Y
    json.dump(run, open(os.path.join(od, 'dec_run.json'), 'w'), indent=1)


def load_new_decoder(cond, s, dev):
    import torch
    dec = new_decoder_module(dev, s)
    dec.load_state_dict(torch.load(os.path.join(OUT, 'dec', f'{S1DATA[cond]}_s{s}.pth'), map_location=dev))
    return dec.eval()


# =============================================================== Sec.2 flows
def show_flow(model, sv, sm, dev, batch=50):
    """all frames shown, process-1 motion visible, process-2 input 0; step t predicts frame t+1.
    -> other features (N, T-1, 64), predicted A-1 images (N, T-1, 16, 64, 3) uint8."""
    torch, R21, util, _ = tenv()
    model.motion_masking = False
    FS, PR = [], []
    with torch.no_grad():
        util.seed_all(0)
        for b0 in range(0, sv.shape[0], batch):
            x = R21.scaled(sv[b0:b0 + batch], dev)
            m = torch.from_numpy(sm[b0:b0 + batch]).float().to(dev)
            model.init_state(x.shape[0])
            fs, pr = [], []
            for t in range(x.shape[1] - 1):
                o = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0, 0)['self_vision']
                fs.append(model.predict_feature(model.get_state()['other'][0]).cpu().numpy())
                pr.append(np.round(((o + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8))
            FS.append(np.stack(fs, 1))
            PR.append(np.stack(pr, 1))
    return np.concatenate(FS), np.concatenate(PR)


def swap_flow(model, D, dev, batch=50):
    """round21_eval.freeze_flow(path 'self', pv 'hold'), V1 (T0 = 23), K = 15, also returning predict_feature(os).
    -> features (N, K, 64), predicted A-1 images (N, K, 16, 64, 3) float32 in [0, 1]."""
    torch, R21, util, _ = tenv()
    N = D['self_vision'].shape[0]
    FS, PR = [], []
    with torch.no_grad():
        for b0 in range(0, N, batch):
            x = R21.scaled(D['self_vision'][b0:b0 + batch], dev)
            m = torch.from_numpy(D['self_motion'][b0:b0 + batch]).float().to(dev)
            B = x.shape[0]
            util.seed_all(0)
            model.init_state(B)
            for t in range(T0):
                model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0, 0)
            P1 = model.compute_pleasure(x[:, T0 - 1])
            hid = torch.ones(B, dtype=torch.bool, device=dev)
            fs, pr = [], []
            for k in range(K):
                o = model({'self_vision': x[:, T0 + k], 'self_motion': m[:, T0 + k]}, 1.0, 1.0, p1_override=P1,
                          p2_override=None, hidden_override=hid)['self_vision']
                fs.append(model.predict_feature(model.get_state()['other'][0]).cpu().numpy())
                pr.append(((o + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
            FS.append(np.stack(fs, 1))
            PR.append(np.stack(pr, 1))
    return np.concatenate(FS), np.concatenate(PR)


def loose_mask(v, colour):
    """v float [0,1] (..., H, W, 3); project convention (channel 0 = 'Red'); thresholds 0.5 (for predicted images)."""
    c0, c1, c2 = v[..., 0], v[..., 1], v[..., 2]
    if colour == 'Red':
        return (c0 > .5) & (c1 < .5) & (c2 < .5)
    return (c1 > .5) & (c0 < .5) & (c2 < .5)


def layout_est(img):
    """img float [0,1] (16, 64, 3) of A-1's view -> layout code (0 ORIGINAL: Red at (-9, +9); 1 SWAP_RG) or -1.
    Red (resp. Green) seen in the upper (y > 0) half of directions means it sits at y = +13 (A-1's |y| <= 10)."""
    for colour, red_up in (('Red', True), ('Green', False)):
        w = loose_mask(img, colour).sum(0).astype(np.float64)
        if w.sum() > 0:
            z = (w * np.exp(1j * COL_ANGLES)).sum()
            if abs(z) > 1e-9 * w.sum():
                up = z.imag > 0
                return int(L.LAYOUT_CODE[L.ORIGINAL] if up == red_up else L.LAYOUT_CODE[L.SWAP_RG])
    return -1


def geom_inputs(imgs, sp, ab, read_a2):
    """imgs float [0,1] (N, T, 16, 64, 3), sp (N, T, 2) A-1 positions -> est (N, T, 2), layout (N, T) int8."""
    N, T = imgs.shape[:2]
    est = np.stack([read_a2(imgs[:, t], sp[:, t], *ab) for t in range(T)], 1)
    lay = np.array([[layout_est(imgs[n, t]) for t in range(T)] for n in range(N)], np.int8)
    return est.astype(np.float32), lay


def s2_main(cond):
    torch, R21, util, read_a2 = tenv()
    import round29 as R29m
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    od = os.path.join(OUT, 's2')
    os.makedirs(od, exist_ok=True)
    ab = R21.calib_common()
    te = load_split('v11_s2_pref', 'test', ('self_vision', 'other_vision', 'self_motion', 'self_position', 'other_position'))
    Pa2_b = pleasure_all(te['other_vision'][:, 1:])                        # truth for frames 1..100
    Dsw, Dns = R21.load_data('swap'), R21.load_data('noswap')
    fr = slice(T0 + 1, T0 + K + 1)                                       # frames 24..38
    Pa2_c, Pa2_c_old = pleasure_all(Dsw['other_vision'][:, fr]), pleasure_all(Dns['other_vision'][:, fr])
    dec0 = existing_decoder(dev).ae_vision_decoder_module
    m0, _ = R21.T.load_v12(EXP['p50'].format(s=0), 0, 200, dev)
    F_a = R29m.enc2(m0, te['self_vision'][:, 1:], dev)                     # real Encoder-2 features of frames 1..100
    F_ca = R29m.enc2(m0, Dsw['self_vision'][:, fr], dev)
    for s in PAIRS:
        t0 = time.time()
        cnn = load_cnn(s, dev)
        model, _ = R21.T.load_v12(EXP[cond].format(s=s), s, 200, dev)
        model.eval()
        F_b, PR_b = show_flow(model, te['self_vision'], te['self_motion'], dev)
        model, _ = R21.T.load_v12(EXP[cond].format(s=s), s, 200, dev)
        model.eval()
        F_c, PR_c = swap_flow(model, Dsw, dev)
        sav = {}
        for dn, dec in (('exist', dec0), ('new', load_new_decoder(cond, s, dev))):
            for sc, F, truth, real in (('a', F_a, Pa2_b, te['other_vision'][:, 1:]), ('b', F_b, Pa2_b, te['other_vision'][:, 1:]),
                                       ('c', F_c, Pa2_c, Dsw['other_vision'][:, fr]), ('ca', F_ca, Pa2_c, Dsw['other_vision'][:, fr])):
                img = decode(dec, F, dev)
                pc = cnn_apply(cnn, img, dev)
                ph = pleasure_all((img * 255).round().astype(np.uint8))
                for k, v in all_suff(pc, truth).items():
                    sav[f'{dn}|{sc}|cnn|{k}'] = v
                for k, v in all_suff(ph, truth).items():
                    sav[f'{dn}|{sc}|hand|{k}'] = v
                if sc in ('c', 'ca'):
                    for k, v in all_suff(pc, Pa2_c_old).items():
                        sav[f'{dn}|{sc}|cnn_old|{k}'] = v
                sav[f'{dn}|{sc}|pix_l1'] = np.abs(img - real.astype(np.float32) / 255.0).mean((-1, -2, -3)).mean(1)
                if s == 0:
                    sav[f'{dn}|{sc}|examples'] = (img[:4, [0, 5, 10, 14] if sc in ('c', 'ca') else [0, 30, 60, 90]] * 255).round().astype(np.uint8)
        if cond == 'p50':
            sp_b = te['self_position'][:, 1:].astype(np.float64)          # A-1 at the predicted frame t + 1
            sav['geo_b_est'], sav['geo_b_lay'] = geom_inputs(PR_b.astype(np.float32) / 255.0, sp_b, ab, read_a2)
            sp_c = Dsw['self_position'][:, T0:T0 + K].astype(np.float64)  # round-21 convention: frame T0 + k - 1
            sav['geo_c_est'], sav['geo_c_lay'] = geom_inputs(PR_c, sp_c, ab, read_a2)
        np.savez_compressed(os.path.join(od, f'{cond}_s{s}.npz'), **sav)
        torch.cuda.empty_cache()
        print(f's2:{cond}:{s} {time.time() - t0:.0f}s {time.strftime("%H:%M")}', flush=True)
    if cond == 'p50':
        np.savez_compressed(os.path.join(od, 'truth.npz'), b=Pa2_b, c=Pa2_c, c_old=Pa2_c_old)


# =============================================================== Sec.3 geometry (CPU, xvfb)
def s3_main(part, n_parts):
    from sim_v10 import creator
    from sim_v10.util import load_config
    import round21_eval as R21
    from p2test_v11 import read_a2
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

    ab = R21.calib_common()
    gb = h5py.File(os.path.join(DATA, 'v11_s2_pref', 'data.h5'), 'r')['test']
    sv_b, sp_b, ov_b = gb['self_vision'][:, 1:], gb['self_position'][:, 1:].astype(np.float64), gb['other_vision'][:, 1:]
    Dsw = R21.load_data('swap')
    fr = slice(T0 + 1, T0 + K + 1)
    sv_c, sp_c_true, ov_c = Dsw['self_vision'][:, fr], Dsw['self_position'][:, fr].astype(np.float64), Dsw['other_vision'][:, fr]
    jobs = [('real_b', None), ('real_c', None)] + [(f'pred_{sc}_s{s}', (sc, s)) for s in PAIRS for sc in ('b', 'c')]
    mine = [j for i, j in enumerate(jobs) if i % n_parts == part]
    od = os.path.join(OUT, 's3')
    os.makedirs(od, exist_ok=True)
    sav = {}
    if part == 0:   # re-render check: A-2's view drawn at its real position / layout equals the stored other_vision
        op_b = gb['other_position'][:, 1:].astype(np.float64)
        bad = sum(int(not np.array_equal(render_other(sp_b[n, t], op_b[n, t], gb['layout'][n]), ov_b[n, t]))
                  for n in range(20) for t in (0, 50, 99))
        sav['rerender_mismatch_b'] = np.array(bad)
        op_c = Dsw['other_position'][:, fr].astype(np.float64)
        bad = sum(int(not np.array_equal(render_other(sp_c_true[n, t], op_c[n, t], Dsw['layout'][n] ^ 1), ov_c[n, t]))
                  for n in range(20) for t in (0, 14))
        sav['rerender_mismatch_c'] = np.array(bad)
        print('rerender mismatches', int(sav['rerender_mismatch_b']), int(sav['rerender_mismatch_c']), flush=True)
    for name, src in mine:
        t0 = time.time()
        if src is None:
            sc = name[-1]
            imgs = (sv_b if sc == 'b' else sv_c).astype(np.float32) / 255.0
            sp = sp_b if sc == 'b' else sp_c_true
            est, lay = geom_inputs(imgs, sp, ab, read_a2)
        else:
            sc, s = src
            z = np.load(os.path.join(OUT, 's2', f'p50_s{s}.npz'))
            est, lay = z[f'geo_{sc}_est'], z[f'geo_{sc}_lay']
        sp_true = sp_b if sc == 'b' else sp_c_true                   # A-1 drawn at its real position of the frame
        ov = ov_b if sc == 'b' else ov_c
        N, T = est.shape[:2]
        P = np.full((N, T, 8), np.nan, np.float32)
        for n in range(N):
            for t in range(T):
                if np.all(np.isfinite(est[n, t])) and lay[n, t] >= 0:
                    q = np.clip(est[n, t], -10, 10)
                    P[n, t] = pleasure_np(render_other(sp_true[n, t], q, lay[n, t]))
        truth = pleasure_all(ov)
        for k, v in all_suff(P, truth).items():
            sav[f'{name}|{k}'] = v
        sav[f'{name}|found_a2'] = np.isfinite(est).all(-1).mean(1)
        sav[f'{name}|found_layout'] = (lay >= 0).mean(1)
        sav[f'{name}|layout_correct'] = np.array([np.mean(lay[n][lay[n] >= 0] == (gb['layout'][n] if sc == 'b' else Dsw['layout'][n] ^ 1))
                                                  if (lay[n] >= 0).any() else np.nan for n in range(N)])
        print(f's3:{name} {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(os.path.join(od, f'part{part}of{n_parts}.npz'), **sav)


# =============================================================== check
def check():
    torch, R21, util, read_a2 = tenv()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    os.makedirs(os.path.join(OUT, 'check'), exist_ok=True)
    out = {}
    # existing decoder: encoders identical to every predictor used
    ae = existing_decoder(dev).state_dict()
    same = []
    for c in CONDS:
        for s in PAIRS:
            m, _ = R21.T.load_v12(EXP[c].format(s=s), s, 200, dev)
            sd = m.state_dict()
            ks = [k for k in ae if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module'))]
            same.append(all(torch.equal(ae[k], sd[k]) for k in ks))
    out['existing_decoder_encoders_identical_18'] = bool(all(same))
    # swap flow reproduces round21_eval.freeze_flow (self, hold) predictions
    D = R21.load_data('swap', 50)
    m, _ = R21.T.load_v12(EXP['p50'].format(s=0), 0, 200, dev)
    ref = R21.freeze_flow(m, D, T0, K, 'self', 'hold', dev)
    m, _ = R21.T.load_v12(EXP['p50'].format(s=0), 0, 200, dev)
    _, pr = swap_flow(m, D, dev)
    out['swap_flow_vs_round21_freeze_flow_max_abs_diff'] = float(np.abs(ref - pr).max())
    # layout estimate on real A-1 views (stage-II test, all frames) and A-1 / A-2 view rendering conventions
    g = h5py.File(os.path.join(DATA, 'v11_s2_pref', 'data.h5'), 'r')['test']
    sv, lay = g['self_vision'][:50], g['layout'][:50]
    le = np.array([[layout_est(sv[n, t].astype(np.float32) / 255.0) for t in range(sv.shape[1])] for n in range(50)])
    out['layout_est_real_found'] = float((le >= 0).mean())
    out['layout_est_real_correct'] = float((le[le >= 0] == np.repeat(lay[:, None], sv.shape[1], 1)[le >= 0]).mean())
    out['shapes'] = {k: list(g[k].shape) for k in ('self_vision', 'other_vision')}
    json.dump(out, open(os.path.join(OUT, 'check', 'check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('s1', 'dec', 's2', 's3', 'check'))
    ap.add_argument('--cond', default='p50')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--n_parts', type=int, default=1)
    a = ap.parse_args()
    {'s1': s1_main, 'dec': dec_main, 'check': check}.get(a.mode, lambda: None)()
    if a.mode == 's2':
        s2_main(a.cond)
    elif a.mode == 's3':
        s3_main(a.part, a.n_parts)
