"""
Paper round 32 (case A; paper/check/paper_instructions_round32.md; docs/paper_extra_log.md Sec.56; exploratory, outside
the pre-registration; procedure fixed before computing). Re-learn a Visual Decoder that draws red and green apart, from
A-1's own views only (input Encoder-1(A-1 view), target A-1 view; same structure and recipe as the existing v12_fig5_s0
except the weight decay / colour-pixel weight), then make the other's pleasure without answers:
A-1 view -> Encoder-2 -> new decoder -> image -> round-30 value net (CNN, Round30/s1/cnn_s<s>.pth) -> red - green.
Only the new decoders are trained; predictors, encoders and the round-30 value nets are never changed.

Modes
  train --recipe D1|D2|D3|D4 --seed s (GPU) decoder on v11_s1_random train episodes 0-999 (the existing decoder's data),
        AdamW lr 1e-3, batch 100, 100 epochs, L1 on the [-1,1] image; D1 wd 0, D2 wd 0.03 (1/100 of 3.0), D3 wd 0 and
        the error of red / green pixels of the target x10, D4 the same x30 (bias never decayed). Validation: train
        episodes 1000-1199 (own views). -> Round32/dec/<recipe>_s<s>.pth, <recipe>_s<s>.json
  select (GPU) existing decoder's validation metrics; recipe choice by the fixed rule -> Round32/dec/select.json
  selfeval (GPU) (a) own-view reconstruction on v11_s1_random test (300 episodes) for every decoder -> Round32/a/
  grid (GPU) (b), (c) on the round-31 grid placements (both layouts, margin 1, overlap flagged) -> Round32/grid/
  flow --cond p50|none|sr50 (GPU) (d) round-30 Sec.2 scenes a / b / c / ca, pairs 0-5 -> Round32/flow/<cond>_s<s>.npz
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
OUT = os.path.join(RES, 'paper_extra', 'Round32')
PAIRS = tuple(range(6))
SEEDS = (0, 1, 2)
RECIPES = {'D1': (0.0, 1.0), 'D2': (0.03, 1.0), 'D3': (0.0, 10.0), 'D4': (0.0, 30.0)}   # (weight decay, colour weight)
EXP = {'p50': 'v12_s1_mixed_l1_s{s}', 'none': 'v12_s1_random_l1_s{s}',
       'sr50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s{s}'}
TRAIN_EP, VAL_EP = (0, 1000), (1000, 1200)
T0, K = 23, 15
C0167 = [0, 1, 6, 7]
EPOCHS = 100


def tenv():
    import torch
    import round21_eval as R21
    import round29 as R29m
    import round30 as R30m
    import round31 as R31m
    return torch, R21, R29m, R30m, R31m


def u8(a):
    return a if a.dtype == np.uint8 else np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)


def pr(x, y, bins=C0167):
    x, y = np.asarray(x)[..., bins].ravel().astype(np.float64), np.asarray(y)[..., bins].ravel().astype(np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def fresh_decoder(dev, seed):
    """an Autoencoder built exactly as the existing one (config v12_fig5_s0), freshly initialised with `seed`."""
    import torch
    from model.model import Autoencoder
    from config_util import load_config
    cfg = load_config(os.path.join(_PI, 'config', 'exp', 'v12_fig5_s0.yml'))
    mcfg = load_config(os.path.join(_PI, 'config', 'model', cfg.model.name, cfg.model.config + '.yml'))
    torch.manual_seed(seed)
    return Autoencoder(mcfg).to(dev).ae_vision_decoder_module


def load_dec(name, dev):
    import torch
    torch_, R21, R29m, R30m, R31m = tenv()
    if name == 'exist':
        return R30m.existing_decoder(dev).ae_vision_decoder_module.eval()
    rec, s = name.split('_s')
    d = fresh_decoder(dev, int(s))
    d.load_state_dict(torch.load(os.path.join(OUT, 'dec', f'{name}.pth'), map_location=dev))
    return d.eval()


def own_metrics(img, V):
    """img float [0,1], V uint8 (N, 16, 64, 3) -> validation / (a) metrics."""
    torch, R21, R29m, R30m, R31m = tenv()
    mh, mv = {}, {}
    for c in range(0, len(V), 2000):
        for d_, x in ((mh, img[c:c + 2000]), (mv, V[c:c + 2000].astype(np.float32) / 255.0)):
            for k, v in R31m.measures(x).items():
                d_.setdefault(k, []).append(v)
    mh = {k: np.concatenate(v) for k, v in mh.items()}
    mv = {k: np.concatenate(v) for k, v in mv.items()}
    return {'l1': float(np.abs(img - V.astype(np.float32) / 255.0).mean()),
            'r_thr_c0167': pr(mh['col_thr'], mv['col_thr']), 'r_thr_all': pr(mh['col_thr'], mv['col_thr'], list(range(8))),
            'r_cont_c0167': pr(mh['col_cont'], mv['col_cont']), 'r_cont_all': pr(mh['col_cont'], mv['col_cont'], list(range(8)))}


def own_views(ep):
    g = h5py.File(os.path.join(DATA, 'v11_s1_random', 'data.h5'), 'r')['train']
    return g['self_vision'][ep[0]:ep[1]].reshape(-1, 16, 64, 3)


# =============================================================== train
def train_main(recipe, seed):
    torch, R21, R29m, R30m, R31m = tenv()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    od = os.path.join(OUT, 'dec')
    os.makedirs(od, exist_ok=True)
    wd, cw = RECIPES[recipe]
    m0, _ = R21.T.load_v12(EXP['p50'].format(s=0), 0, 200, dev)
    m0.eval()
    Vtr, Vva = own_views(TRAIN_EP), own_views(VAL_EP)
    F = torch.from_numpy(R29m.enc1(m0, Vtr, dev)).to(dev)
    Fva = R29m.enc1(m0, Vva, dev)
    Y = torch.from_numpy(Vtr).to(dev)
    c = Y.float() / 255.0
    col = (((c[..., 0] > .9) & (c[..., 1] < .1) & (c[..., 2] < .1)) | ((c[..., 1] > .9) & (c[..., 0] < .1) & (c[..., 2] < .1)))
    Wpix = (1.0 + (cw - 1.0) * col.float()).unsqueeze(1)          # (N, 1, 16, 64)
    del c
    dec = fresh_decoder(dev, seed).train()
    decay = [p for n, p in dec.named_parameters() if not n.endswith('bias')]
    nodecay = [p for n, p in dec.named_parameters() if n.endswith('bias')]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': wd}, {'params': nodecay, 'weight_decay': 0.0}], lr=1e-3)
    g = torch.Generator(device='cpu').manual_seed(seed)
    N = F.shape[0]
    log = []
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(N, generator=g).to(dev)
        tot, tot_u = 0.0, 0.0
        for b0 in range(0, N, 100):
            ib = perm[b0:b0 + 100]
            tgt = (Y[ib].float() / 255.0 * 2 - 1).permute(0, 3, 1, 2)
            err = (dec(F[ib]) - tgt).abs()
            loss = (err * Wpix[ib]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(ib)
            tot_u += float(err.mean()) * len(ib)
        log.append([tot / N, tot_u / N])
        if ep % 10 == 9:
            print(f'train:{recipe}_s{seed} ep{ep + 1} loss {tot / N:.4f} l1 {tot_u / N:.4f} {time.time() - t0:.0f}s', flush=True)
    dec.eval()
    torch.save(dec.state_dict(), os.path.join(od, f'{recipe}_s{seed}.pth'))
    val = own_metrics(R30m.decode(dec, Fva, dev), Vva)
    json.dump({'recipe': recipe, 'seed': seed, 'weight_decay': wd, 'colour_weight': cw, 'train_log': log, 'val': val,
               'colour_pixel_frac_train': float(col.float().mean()), 'sec': time.time() - t0},
              open(os.path.join(od, f'{recipe}_s{seed}.json'), 'w'), indent=1)
    print(f'train:{recipe}_s{seed} done val {val}', flush=True)


# =============================================================== select
def select_main():
    """rule (Sec.56.2): among recipes whose mean validation L1 (3 seeds) <= 1.5 x the existing decoder's, the highest
    mean (3 seeds) validation r of the thresholded red - green (ranges 0, 1, 6, 7); undefined r counts as -inf; if all
    are undefined, the continuous r decides."""
    torch, R21, R29m, R30m, R31m = tenv()
    dev = torch.device('cuda:0')
    m0, _ = R21.T.load_v12(EXP['p50'].format(s=0), 0, 200, dev)
    Vva = own_views(VAL_EP)
    ex = own_metrics(R30m.decode(load_dec('exist', dev), R29m.enc1(m0, Vva, dev), dev), Vva)
    rows = {}
    for r in RECIPES:
        v = [json.load(open(os.path.join(OUT, 'dec', f'{r}_s{s}.json')))['val'] for s in SEEDS]
        m = lambda k: float(np.mean([x[k] for x in v])) if all(np.isfinite(x[k]) for x in v) else float('nan')  # noqa
        rows[r] = {'per_seed': v, 'l1': m('l1'), 'r_thr_c0167': m('r_thr_c0167'), 'r_cont_c0167': m('r_cont_c0167'),
                   'l1_ok': bool(m('l1') <= 1.5 * ex['l1'])}
    ok = [r for r in RECIPES if rows[r]['l1_ok']]
    key = 'r_thr_c0167' if any(np.isfinite(rows[r]['r_thr_c0167']) for r in ok) else 'r_cont_c0167'
    sc = {r: (rows[r][key] if np.isfinite(rows[r][key]) else -np.inf) for r in ok}
    chosen = max(sc, key=sc.get) if sc else None
    out = {'existing_val': ex, 'recipes': rows, 'eligible': ok, 'key': key, 'chosen': chosen}
    json.dump(out, open(os.path.join(OUT, 'dec', 'select.json'), 'w'), indent=1)
    print(json.dumps({'existing': ex, 'chosen': chosen, 'key': key,
                      'rows': {r: {k: rows[r][k] for k in ('l1', 'r_thr_c0167', 'r_cont_c0167', 'l1_ok')} for r in rows}}, indent=1))


def decoder_names(all_new=False):
    ch = json.load(open(os.path.join(OUT, 'dec', 'select.json')))['chosen']
    if ch is None:      # no recipe eligible: the existing decoder and every recipe's seed 0, descriptive only
        return None, ['exist'], [f'{r}_s0' for r in RECIPES]
    names = ['exist'] + [f'{ch}_s{s}' for s in SEEDS]
    others = [f'{r}_s{s}' for r in RECIPES if r != ch for s in (SEEDS if all_new else (0,))]
    return ch, names, others


# =============================================================== (a) own views
def selfeval_main():
    torch, R21, R29m, R30m, R31m = tenv()
    dev = torch.device('cuda:0')
    od = os.path.join(OUT, 'a')
    os.makedirs(od, exist_ok=True)
    ch, names, others = decoder_names(all_new=True)
    m0, _ = R21.T.load_v12(EXP['p50'].format(s=0), 0, 200, dev)
    V = h5py.File(os.path.join(DATA, 'v11_s1_random', 'data.h5'), 'r')['test']['self_vision'][()]   # (300, 101, ...)
    N, T = V.shape[:2]
    F1 = R29m.enc1(m0, V, dev)
    mv = {}
    flat = V.reshape(-1, 16, 64, 3)
    for c in range(0, len(flat), 3030):
        for k, v in R31m.measures(flat[c:c + 3030].astype(np.float32) / 255.0).items():
            mv.setdefault(k, []).append(v)
    mv = {k: np.concatenate(v).reshape(N, T, 8) for k, v in mv.items()}
    sav = {}
    for dn in names + others:
        dec = load_dec(dn, dev)
        img = R30m.decode(dec, F1, dev)                                          # (N, T, 16, 64, 3)
        mh = {}
        fl = img.reshape(-1, 16, 64, 3)
        for c in range(0, len(fl), 3030):
            for k, v in R31m.measures(fl[c:c + 3030]).items():
                mh.setdefault(k, []).append(v)
        mh = {k: np.concatenate(v).reshape(N, T, 8) for k, v in mh.items()}
        for mk in ('col_thr', 'col_cont'):
            for k, v in R30m.all_suff(mh[mk], mv[mk]).items():
                sav[f'{dn}|{mk}|{k}'] = v
        sav[f'{dn}|pix_l1'] = np.abs(img - V.astype(np.float32) / 255.0).mean((-1, -2, -3)).mean(1)
        print(f'a:{dn} {time.strftime("%H:%M")}', flush=True)
    np.savez_compressed(os.path.join(od, 'a.npz'), **sav)


# =============================================================== (b), (c) grid
def grid_main():
    torch, R21, R29m, R30m, R31m = tenv()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    od = os.path.join(OUT, 'grid')
    os.makedirs(od, exist_ok=True)
    ch, names, others = decoder_names()
    decs = {dn: load_dec(dn, dev) for dn in names + others}
    cnns = [R30m.load_cnn(s, dev) for s in PAIRS]
    m0, _ = R21.T.load_v12(EXP['p50'].format(s=0), 0, 200, dev)
    KEEP2 = R31m.KEEP2
    D = {}
    ex = {}

    def add(k, v):
        D.setdefault(k, []).append(v)

    for li, (lay, path) in enumerate(R31m.GRIDS.items()):
        g = h5py.File(path, 'r')['train']
        t0 = time.time()
        for o in KEEP2:
            sv, ov = u8(g['self_vision'][o][KEEP2]), u8(g['other_vision'][o][KEEP2])
            v1, v2 = sv.astype(np.float32) / 255.0, ov.astype(np.float32) / 255.0
            F1, F2 = R29m.enc1(m0, sv, dev), R29m.enc2(m0, sv, dev)
            add('layout', np.full(len(KEEP2), li, np.int8))
            add('o', np.full(len(KEEP2), o, np.int16))
            add('overlap', KEEP2 == o)
            for nm, img in (('v1', v1), ('v2', v2)):
                mm = R31m.measures(img)
                for mk in ('col_thr', 'col_cont', 'pos_thr', 'pos_cont'):
                    add(f'{nm}_{mk}', mm[mk].astype(np.float16))
            for s, net in enumerate(cnns):
                add(f'v2_cnn{s}', R30m.cnn_apply(net, ov, dev).astype(np.float16))      # value net on the real A-2 view
            for dn, dec in decs.items():
                h1, h2 = R30m.decode(dec, F1, dev), R30m.decode(dec, F2, dev)
                dd = lambda a, b: np.abs(a - b).mean((1, 2, 3)).astype(np.float32)  # noqa
                add(f'{dn}|d_h2_v2', dd(h2, v2))
                add(f'{dn}|d_h2_v1', dd(h2, v1))
                add(f'{dn}|d_h1_v1', dd(h1, v1))
                add(f'{dn}|d_h1_v2', dd(h1, v2))
                m2, m1 = R31m.measures(h2), R31m.measures(h1)
                for mk in ('col_thr', 'col_cont', 'pos_thr', 'pos_cont'):
                    add(f'{dn}|h2_{mk}', m2[mk].astype(np.float16))
                for mk in ('col_thr', 'col_cont'):
                    add(f'{dn}|h1_{mk}', m1[mk].astype(np.float16))
                for s, net in enumerate(cnns):
                    add(f'{dn}|h2_cnn{s}', R30m.cnn_apply(net, h2, dev).astype(np.float16))
                for a1, a2 in R31m.EX_CONFIGS:
                    if (a2[0] + 10) * 21 + (a2[1] + 10) == o:
                        j = int(np.where(KEEP2 == (a1[0] + 10) * 21 + (a1[1] + 10))[0][0])
                        ex.setdefault(f'{lay}|A1{a1}|A2{a2}|v', np.stack([sv[j], ov[j]]))
                        ex[f'{lay}|A1{a1}|A2{a2}|{dn}'] = np.stack([u8(h1[j]), u8(h2[j])])
        print(f'grid:{lay} {time.time() - t0:.0f}s {time.strftime("%H:%M")}', flush=True)
    np.savez(os.path.join(od, 'grid.npz'), **{k: np.concatenate(v) for k, v in D.items()})
    np.savez_compressed(os.path.join(od, 'examples.npz'), **ex)
    print('grid done', flush=True)


# =============================================================== (d) flows
def flow_main(cond):
    torch, R21, R29m, R30m, R31m = tenv()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    od = os.path.join(OUT, 'flow')
    os.makedirs(od, exist_ok=True)
    ch, names, _ = decoder_names()
    decs = {dn: load_dec(dn, dev) for dn in names}
    te = R30m.load_split('v11_s2_pref', 'test', ('self_vision', 'other_vision', 'self_motion'))
    Pa2_b = R30m.pleasure_all(te['other_vision'][:, 1:])
    Dsw, Dns = R21.load_data('swap'), R21.load_data('noswap')
    fr = slice(T0 + 1, T0 + K + 1)
    Pa2_c, Pa2_c_old = R30m.pleasure_all(Dsw['other_vision'][:, fr]), R30m.pleasure_all(Dns['other_vision'][:, fr])
    m0, _ = R21.T.load_v12(EXP['p50'].format(s=0), 0, 200, dev)
    F_a = R29m.enc2(m0, te['self_vision'][:, 1:], dev)
    F_ca = R29m.enc2(m0, Dsw['self_vision'][:, fr], dev)
    for s in PAIRS:
        t0 = time.time()
        cnn = R30m.load_cnn(s, dev)
        model, _ = R21.T.load_v12(EXP[cond].format(s=s), s, 200, dev)
        model.eval()
        F_b, _ = R30m.show_flow(model, te['self_vision'], te['self_motion'], dev)
        model, _ = R21.T.load_v12(EXP[cond].format(s=s), s, 200, dev)
        model.eval()
        F_c, _ = R30m.swap_flow(model, Dsw, dev)
        sav = {}
        for dn, dec in decs.items():
            for sc, Fx, truth, real in (('a', F_a, Pa2_b, te['other_vision'][:, 1:]), ('b', F_b, Pa2_b, te['other_vision'][:, 1:]),
                                        ('c', F_c, Pa2_c, Dsw['other_vision'][:, fr]), ('ca', F_ca, Pa2_c, Dsw['other_vision'][:, fr])):
                img = R30m.decode(dec, Fx, dev)
                pc = R30m.cnn_apply(cnn, img, dev)
                ph = R30m.pleasure_all(u8(img))
                for k, v in R30m.all_suff(pc, truth).items():
                    sav[f'{dn}|{sc}|cnn|{k}'] = v
                for k, v in R30m.all_suff(ph, truth).items():
                    sav[f'{dn}|{sc}|hand|{k}'] = v
                if sc in ('c', 'ca'):
                    for k, v in R30m.all_suff(pc, Pa2_c_old).items():
                        sav[f'{dn}|{sc}|cnn_old|{k}'] = v
                sav[f'{dn}|{sc}|pix_l1'] = np.abs(img - real.astype(np.float32) / 255.0).mean((-1, -2, -3)).mean(1)
                if s == 0 and cond == 'p50':
                    fi = [0, 5, 10, 14] if sc in ('c', 'ca') else [0, 30, 60, 90]
                    sav[f'{dn}|{sc}|examples'] = u8(img[:3, fi])
                    sav[f'real|{sc}|examples'] = real[:3, fi]
        np.savez_compressed(os.path.join(od, f'{cond}_s{s}.npz'), **sav)
        torch.cuda.empty_cache()
        print(f'flow:{cond}:{s} {time.time() - t0:.0f}s {time.strftime("%H:%M")}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('train', 'select', 'selfeval', 'grid', 'flow'))
    ap.add_argument('--recipe', default='D1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cond', default='p50')
    a = ap.parse_args()
    if a.mode == 'train':
        train_main(a.recipe, a.seed)
    elif a.mode == 'flow':
        flow_main(a.cond)
    else:
        {'select': select_main, 'selfeval': selfeval_main, 'grid': grid_main}[a.mode]()
