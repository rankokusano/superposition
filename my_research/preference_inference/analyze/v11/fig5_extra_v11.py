"""
v11 additional (non-pre-registered) Fig.5-reason checks (docs/v11_experiment_log.md
Sec.7.44; methods recorded before computing). CPU only.
  check1: read-out of A-2's true view (top-50 PCs) from (a) ov_enc, (b) sv_enc,
          (c) A-1 view pixels (top-100 PCs), (d) A-2 position x layout (poly-3),
          (d') + A-1 position; Ridge(1), episode GroupKFold(5).
  check2: linear map W: ov(v_A1) -> sv(v_A2) (Ridge(1)+bias, ReLU), fit on train
          frames, test R^2 / cosine; Fig.5 margin and map minimum with ReLU(W ov + b)
          through each model's Fig.5 decoder.
  check3: alignment cos(ov(v_A1), sv(v_A2)) vs cos(ov(v_A1), sv(v_A1)) for v5 and
          v10 B3-frozen / B3-ft (read-only checkpoints) on their own environments.
Encoders are loaded from any checkpoint by key (self/other_vision_encoder_module,
share_lns) into the paper's Autoencoder class.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import DATA, RES  # noqa
import util  # noqa
from config_util import load_config  # noqa
from model.model import Autoencoder  # noqa

_PI = os.path.dirname(os.path.dirname(_HERE))
MAIN = '/work/my_research/preference_inference/data'
ROOT_RES = '/work/data/result'
STRIDE = 5
SIZE = 21


def make_ae(ckpt, ae_ckpt=None):
    cfg = load_config(os.path.join(_PI, 'config', 'model', 'Autoencoder', 'default.yml'))
    m = Autoencoder(cfg)
    sd = torch.load(ckpt, map_location='cpu')['model']
    keep = {k: v for k, v in sd.items() if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module', 'share_lns'))}
    if ae_ckpt is not None:
        sd2 = torch.load(ae_ckpt, map_location='cpu')['model']
        keep.update({k: v for k, v in sd2.items() if k.startswith('ae_vision_decoder_module')})
    missing = m.load_state_dict(keep, strict=False)
    m.eval()
    return m, missing


@torch.no_grad()
def enc(module, frames, bs=2000):
    out = []
    for i in range(0, len(frames), bs):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(frames[i:i + bs], dim=4)))
        out.append(module(x).numpy())
    return np.concatenate(out)


def frames(path, split, eps=None, layout_code=None, n_max=None):
    g = h5py.File(path, 'r')[split]
    n = g['self_vision'].shape[0] if n_max is None else n_max
    idx = np.arange(n) if eps is None else eps
    if layout_code is not None and 'layout' in g:
        idx = idx[g['layout'][()][idx] == layout_code]
    ts = np.arange(0, 100, STRIDE)
    sv = g['self_vision'][()][idx][:, ts]
    ov = g['other_vision'][()][idx][:, ts]
    sp = g['self_position'][()][idx][:, ts]
    op = g['other_position'][()][idx][:, ts]
    lay = g['layout'][()][idx] if 'layout' in g else np.zeros(len(idx), int)
    grp = np.repeat(np.arange(len(idx))[:, None], len(ts), 1).reshape(-1)
    f = lambda a: a.reshape((-1,) + a.shape[2:])  # noqa
    return f(sv), f(ov), f(sp), f(op), np.repeat(lay[:, None], len(ts), 1).reshape(-1), grp


def pcs(V, k):
    V = V.reshape(len(V), -1).astype(np.float32) / 255.0
    V = V - V.mean(0)
    _, _, vt = np.linalg.svd(V, full_matrices=False)
    return V @ vt[:k].T


def cv_r2(X, Y, g):
    Yh = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(X, Y, g):
        Yh[te] = Ridge(alpha=1.0).fit(X[tr], Y[tr]).predict(X[te])
    return float(1 - ((Y - Yh) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def cosm(a, b):
    return float(((a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)).mean())


def rel(a, b):
    return float(((a - b) ** 2).sum() / ((b - b.mean(0)) ** 2).sum())


def v11_models():
    out = []
    for c in ('random', 'mixed'):
        for s in (0, 1, 2):
            out.append((f'v11_{c}_s{s}', os.path.join(RES, f'v11_s1_{c}_l1_s{s}', str(s), 'model', '00200.pth'),
                        os.path.join(RES, f'v11_fig5_{c}_s{s}', str(s), 'model', '00100.pth')))
    out.append(('v4_paper', os.path.join(ROOT_RES, 'exp1_l1_1000', '0', 'model', '00200.pth'),
                os.path.join(MAIN, 'result', 'fig5_v4_exp1l1000', '0', 'model', '00100.pth')))
    return out


def check1(res):
    test = os.path.join(DATA, 'v11_s1_random', 'data.h5')
    for name, ck, _ in v11_models():
        m, _ = make_ae(ck)
        for sub, lc in (('original_only', 0), ('both_layouts', None)):
            if name.startswith('v4') and lc is None:
                continue
            sv, ov, sp, op, lay, grp = frames(test, 'test', layout_code=lc)
            Y = pcs(ov, 50)
            OE, SE = enc(m.other_vision_encoder_module, sv), enc(m.self_vision_encoder_module, sv)
            pix = pcs(sv, 100)
            pf = PolynomialFeatures(3)
            onehot = np.eye(2)[lay]
            d = np.concatenate([pf.fit_transform(op) * onehot[:, [k]] for k in (0, 1)], 1)
            d2 = np.concatenate([d] + [pf.fit_transform(sp) * onehot[:, [k]] for k in (0, 1)], 1)
            r = {'a_ov_enc': cv_r2(OE, Y, grp), 'b_sv_enc': cv_r2(SE, Y, grp), 'c_A1_pixels_pc100': cv_r2(pix, Y, grp),
                 'd_A2pos_x_layout': cv_r2(d, Y, grp), 'd2_plus_A1pos': cv_r2(d2, Y, grp)}
            res.setdefault('check1', {})[f'{name}/{sub}'] = r
            print('check1', name, sub, {k: round(v, 3) for k, v in r.items()}, flush=True)


@torch.no_grad()
def fig5_with_map(m, W, b, grid_path, margin=1):
    g = h5py.File(grid_path, 'r')['train']
    f32 = lambda a: a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)  # noqa
    true_images = f32(g['self_vision_no_agent'][0])
    keep_1d = np.arange(margin, SIZE - margin)
    keep = (keep_1d[:, None] * SIZE + keep_1d[None, :]).reshape(-1)
    oo, os_, hit, n = [], [], 0, 0
    if W is not None:
        Wt, bt = torch.from_numpy(W.T.astype(np.float32)), torch.from_numpy(b.astype(np.float32))
    for o in keep:
        sv = f32(g['self_vision'][o][keep])
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(sv, dim=4)))
        e = m.other_vision_encoder_module(x)
        e = torch.relu(e @ Wt + bt) if W is not None else e
        rec = ((m.ae_vision_decoder_module(e) + 1) / 2).permute(0, 2, 3, 1).numpy()
        ot = f32(g['other_vision_no_agent'][o][keep])
        st = f32(g['self_vision_no_agent'][o][keep])
        oo.append(np.abs(ot - rec).mean((1, 2, 3)))
        os_.append(np.abs(st - rec).mean((1, 2, 3)))
        for j in range(0, len(keep), 4):
            em = np.abs(true_images - rec[j]).mean((1, 2, 3))
            p = np.array(divmod(int(np.argmin(em)), SIZE))
            hit += int(np.abs(p - np.array(divmod(int(o), SIZE))).max() <= 1)
            n += 1
    oo, os_ = np.concatenate(oo), np.concatenate(os_)
    # margin as in Fig.5: other_self - other_other with other_self = |A-2 view - self_rec|; here we report
    # |A-2 view - mapped other_rec| (other_other) and |A-1 view - mapped other_rec| (self_other)
    return {'other_other': float(oo.mean()), 'self_other': float(os_.mean()),
            'self_other_minus_other_other': float(os_.mean() - oo.mean()), 'map_min_within1_of_A2': hit / n}


def check2(res):
    train = os.path.join(DATA, 'v11_s1_random', 'data.h5')
    grids = {'original': os.path.join(DATA, 'grid', 'data.h5'), 'swap_rg': os.path.join(DATA, 'v10_swap_grid', 'data.h5')}
    for name, ck, ae in v11_models():
        m, _ = make_ae(ck, ae)
        lc = 0 if name.startswith('v4') else None
        sv, ov, *_ = frames(train, 'train', eps=np.arange(500), layout_code=lc)
        X, Y = enc(m.other_vision_encoder_module, sv), enc(m.self_vision_encoder_module, ov)
        reg = Ridge(alpha=1.0).fit(X, Y)
        svt, ovt, *_ = frames(train.replace('train', 'train'), 'test', layout_code=lc)
        Xt, Yt = enc(m.other_vision_encoder_module, svt), enc(m.self_vision_encoder_module, ovt)
        Yh = np.maximum(reg.predict(Xt), 0)
        r = {'map_test_r2': float(1 - ((Yt - Yh) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum()),
             'cos_before': cosm(Xt, Yt), 'cos_after': cosm(Yh, Yt)}
        for gname, gp in grids.items():
            if name.startswith('v4') and gname == 'swap_rg':
                continue
            r[f'fig5_mapped_{gname}'] = fig5_with_map(m, reg.coef_, reg.intercept_, gp)
            r[f'fig5_unmapped_{gname}'] = fig5_with_map(m, None, None, gp)
        res.setdefault('check2', {})[name] = r
        print('check2', name, json.dumps(r), flush=True)


def check3(res):
    specs = [('v5_base_l1_s0', os.path.join(MAIN, 'result', 'v5_base_l1', '0', 'model', '00200.pth'),
              os.path.join(MAIN, 'data', 'r3_stay', 'data.h5'))]
    for kind in ('frozen', 'ft'):
        for s in (0, 1, 2):
            specs.append((f'v10_b3_{kind}_s{s}', os.path.join(MAIN, 'result', f'v10_b3_{kind}', str(s), 'model', '00400.pth'),
                          os.path.join(MAIN, 'data', 'v10_swapmix_r2', 'data.h5')))
    for name, ck, data in specs:
        m, _ = make_ae(ck)
        sv, ov, *_ = frames(data, 'test')
        ov_a1 = enc(m.other_vision_encoder_module, sv)
        sv_a2 = enc(m.self_vision_encoder_module, ov)
        sv_a1 = enc(m.self_vision_encoder_module, sv)
        r = {'cos_ov(A1)_sv(A2)': cosm(ov_a1, sv_a2), 'cos_ov(A1)_sv(A1)': cosm(ov_a1, sv_a1),
             'rel_ov(A1)_vs_sv(A2)': rel(ov_a1, sv_a2), 'rel_ov(A1)_vs_sv(A1)': rel(ov_a1, sv_a1), 'data': data}
        res.setdefault('check3', {})[name] = r
        print('check3', name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checks', nargs='+', default=['3', '1', '2'])
    args = ap.parse_args()
    torch.set_num_threads(8)
    out = os.path.join(RES, 'v11', 'main', 'fig5_reason')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'fig5_extra.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    for c in args.checks:
        {'1': check1, '2': check2, '3': check3}[c](res)
        with open(os.path.join(out, 'fig5_extra.json'), 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
