"""
Why are red and green yellowish in the Fig.5 views? (2026-10-23, Kusano-san's question; no training, CPU.)
For several Fig.5 autoencoders (encoders + Fig.5 Decoder), show and quantify
  * self reconstruction  D(E1(v_A1))  vs A-1's true view, and
  * other view           D(E2(v_A1))  vs A-2's true view (Fig.5),
on v11_s1_random test (both layouts) and, for v4, also on its own environment (self_random_other_stay_1000, no swap).
Colour separation index on the true red / green landmark pixels: sep = mean(R - G | true red) - mean(R - G | true green)
of the reconstruction in [0, 1] (true views give about 2; a yellow mixture gives about 0; negative = colours swapped).
AEs: v4 (v4 encoders, Decoder trained in the v4 environment, `fig5_v4_exp1l1000`; read only from the root checkout),
v12 (v4 encoders, Decoder trained in the swapping environment, `v12_fig5_s0`), E1 mse ep100 (`v12_e1_fig5_mse100`),
v11 random s0 (`v11_fig5_random_s0`, reference). Writes data/result/v12/fig5_colour/{fig5_colour.png,json}.
"""

import json
import os
import sys

import h5py
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from fig5_extra_v11 import make_ae  # noqa
from eval_trial_v11 import DATA, RES  # noqa
import util  # noqa

AES = [('v4（v4 の Encoder・v4 の環境で学習した Decoder）', '/work/my_research/preference_inference/data/result/fig5_v4_exp1l1000/0/model/00100.pth'),
       ('v12（v4 の Encoder を凍結・入れ替えのある環境で学習した Decoder）', os.path.join(RES, 'v12_fig5_s0', '0', 'model', '00100.pth')),
       ('E1 ep100（運動だけ・入れ替えあり・経験なし、一から）', os.path.join(RES, 'v12_e1_fig5_mse100', '0', 'model', '00100.pth')),
       ('参考：v11 経験なし s0（快・不快だけ・入れ替えあり、一から）', os.path.join(RES, 'v11_fig5_random_s0', '0', 'model', '00100.pth'))]
V4ENV = '/work/data/data/self_random_other_stay_1000/data.h5'


@torch.no_grad()
def rec(m, frames, other=False):
    out = []
    for i in range(0, len(frames), 1000):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(frames[i:i + 1000], dim=4)))
        mod = m.other_vision_encoder_module if other else m.self_vision_encoder_module
        y = m.ae_vision_decoder_module(mod(x))
        out.append(((y + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy())
    return np.concatenate(out)


def masks(true):
    t = true.astype(np.int16)
    red = (t[..., 0] > 150) & (t[..., 1] < 80) & (t[..., 2] < 80)
    green = (t[..., 1] > 150) & (t[..., 0] < 80) & (t[..., 2] < 80)
    return red, green


def sep(recon, true):
    red, green = masks(true)
    d = recon[..., 0] - recon[..., 1]
    return float(d[red].mean() - d[green].mean()), float(d[red].mean()), float(d[green].mean())


def main():
    torch.set_num_threads(8)
    out = os.path.join(RES, 'v12', 'fig5_colour'); os.makedirs(out, exist_ok=True)
    g = h5py.File(os.path.join(DATA, 'v11_s1_random', 'data.h5'), 'r')['test']
    ts = np.arange(0, 100, 5)
    lay = g['layout'][()]
    sv = g['self_vision'][()][:, ts]; ov = g['other_vision'][()][:, ts]
    g4 = h5py.File(V4ENV, 'r')['test']
    sv4 = g4['self_vision'][()][:, ts].reshape(-1, 16, 64, 3)
    if sv4.dtype != np.uint8:   # the v4 environment data are float32 in [0, 1]; same input after scale_vision
        sv4 = np.round(sv4 * 255).astype(np.uint8)
    # example frames: one per layout where both views show red and green
    ex = []
    for L in (0, 1):
        best = None
        for e in np.where(lay == L)[0]:
            for k in range(len(ts)):
                r1, g1 = masks(sv[e, k][None]); r2, g2 = masks(ov[e, k][None])
                score = min(r1.sum(), g1.sum(), r2.sum(), g2.sum())
                if best is None or score > best[0]:
                    best = (score, e, k)
        ex.append(best[1:])
    e4 = int(np.argmax([min(m.sum() for m in masks(f[None])) for f in sv4]))
    res = {}
    rows = []
    for name, ck in AES:
        m, _ = make_ae(ck, ck)
        r = {}
        for L, ln in ((0, 'original'), (1, 'swap_rg')):
            sel = lay == L
            S = sv[sel].reshape(-1, 16, 64, 3); O = ov[sel].reshape(-1, 16, 64, 3)
            r[ln] = {'self_rec_sep': sep(rec(m, S), S), 'other_view_sep': sep(rec(m, S, True), O), 'true_view_sep': sep(S / 255.0, S)}
        r['v4_env_self_rec_sep'] = sep(rec(m, sv4), sv4)
        res[name] = r
        imgs = []
        for e, k in ex:
            f = sv[e, k][None]
            imgs += [f[0] / 255, rec(m, f)[0], ov[e, k] / 255, rec(m, f, True)[0]]
        f4 = sv4[e4][None]
        imgs += [f4[0] / 255, rec(m, f4)[0], None, rec(m, f4, True)[0]]
        rows.append((name, imgs))
        print(name, json.dumps({k: (v if not isinstance(v, dict) else {kk: [round(x, 3) for x in vv] for kk, vv in v.items()})
                                 for k, v in r.items()}), flush=True)
    json.dump(res, open(os.path.join(out, 'fig5_colour.json'), 'w'), indent=1, ensure_ascii=False)
    cols = ['A-1 の視界（正解）', '自分側の再構成', 'A-2 の視界（正解）', '他者側の視点（Fig.5）']
    heads = ['元の配置（v11 の test）', '入れ替えた配置（v11 の test）', 'v4 の環境（入れ替えなし）']
    from matplotlib import font_manager
    if os.path.exists('/tmp/jpfont.ttc'):
        font_manager.fontManager.addfont('/tmp/jpfont.ttc')
        plt.rcParams['font.family'] = font_manager.FontProperties(fname='/tmp/jpfont.ttc').get_name()
    fig, axes = plt.subplots(len(rows), 12, figsize=(26, 1.55 * len(rows) + 1.2),
                             gridspec_kw={'wspace': 0.06, 'hspace': 0.55})
    keys = [('original', 'self_rec_sep'), ('original', 'other_view_sep'), ('swap_rg', 'self_rec_sep'), ('swap_rg', 'other_view_sep')]
    for i, (name, imgs) in enumerate(rows):
        r = res[name]
        for j, im in enumerate(imgs):
            ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
            if im is None:
                ax.axis('off'); ax.text(0.5, 0.5, '（データに A-2 の\n視界がない）', ha='center', va='center', fontsize=8, transform=ax.transAxes)
            else:
                ax.imshow(im, interpolation='nearest', aspect='auto')
            blk, c = j // 4, j % 4
            if c in (1, 3) and im is not None:
                if blk < 2:
                    v = r[('original', 'swap_rg')[blk]]['self_rec_sep' if c == 1 else 'other_view_sep'][0]
                elif c == 1:
                    v = r['v4_env_self_rec_sep'][0]
                else:
                    v = None
                if v is not None:
                    ax.set_xlabel(f'sep {v:+.2f}', fontsize=9, color=('#1a7f37' if v > 0.8 else '#b3261e' if v < -0.8 else '#8a6d00'))
            if i == 0:
                ax.set_title(f'{heads[blk]}\n{cols[c]}', fontsize=8)
        axes[i, 0].set_ylabel(name.replace('（', '\n（'), fontsize=8, rotation=0, ha='right', va='center')
    fig.suptitle('Fig.5 の色：自分側の再構成 D(E1(v)) と他者側の視点 D(E2(v))。sep = 赤の画素の (R−G) − 緑の画素の (R−G)（正解の視界で約 +1.94、黄色に混ざると 0、色が逆だと負）。'
                 '配列のチャンネルをそのまま RGB で表示（v10 の約束）', fontsize=10)
    fig.subplots_adjust(left=0.165, right=0.995, top=0.86, bottom=0.06)
    fig.savefig(os.path.join(out, 'fig5_colour.png'), dpi=110)
    print('saved', out)


if __name__ == '__main__':
    main()
