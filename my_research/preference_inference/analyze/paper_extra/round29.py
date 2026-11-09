"""
Paper round 29 (paper/check/paper_instructions_round29.md; docs/paper_extra_log.md Sec.53; exploratory, outside the
pre-registration; the procedure below is fixed before computing). Checks before building the new mechanism
(frequency-based preference estimate + own pleasure computation for the other + motion model into process-2's motion
slot). The predictors are never changed; the only training is the small read-outs of Sec.2 / Sec.3.

Modes
  views  --part i --n_parts n   (CPU, xvfb) Sec.1 reference views: round 21 data_noswap (A-2 static), A-2 at
         p2 + k BIN_UNIT[b] (b = 0..7, k = 1..15) at frame 23 + k, A-1 at its real position of that frame, the episode's
         layout; valid = inside the arena (|x|, |y| <= 10). Re-render check: A-2 at p2 at frames 24..38 equals the data.
         -> Round29/views/views_part<i>of<n>.h5
  encode (GPU) Encoder-2 (other_vision_encoder, identical in every predictor, checked) of the views -> Round29/ref/
  s1 --worker w --n_workers n (GPU) Sec.1: flow of round 23 Sec.2.2 (load_v12, frames 0..22 shown, freeze from T0 = 23,
         p1 = 0 and process-1 motion hidden), K = 15 steps; process-2 input p2 = [m (2), 0 x 8, flag 0] with m = BIN_UNIT[b]
         (b = 0..7) at every frozen step, or m = 0 (zero input = the stage-I process-2 input). p50 / none / sr50 / p80,
         pairs 0-5, ep160-200. Per input: img slopes and feat scores along all 8 directions -> Round29/s1/<c>_s<s>_ep<e>.npz
  s2 (CPU) Sec.2: pleasure(t) -> sector of A-1's step t (self_motion[t], the step from frame t to t+1) on the stage-I
         data of p50 / p80 / none / sr50 (sr50: sign per episode as seen by that predictor); MLP 8-32-8 and linear 8-8,
         6 seeds; chance / frequency / best (rule oracle) levels; reaction to +0.25 in one range -> Round29/s2/
  s3 (GPU) Sec.3: ridge read-out Encoder-1(A-1 view) -> A-1's red - green (8 ranges) learnt on the condition's stage-I
         train split; applied to Encoder-1(A-2's real view), Encoder-2(A-1 view) (descriptive) and the predicted
         other-view features predict_feature(os) (flow 'show' = all frames shown, main; 'freeze' = round-23 freeze,
         descriptive), compared with the red - green of A-2's view. p50 / none / sr50, pairs 0-5, ep200 -> Round29/s3/
  s4 (CPU) Sec.4: ideal observer of A-2's liked colour (rule known, positions known): log likelihood ratio per step
         with direction bins of 5 deg (main) and 45 deg (descriptive); v11_s2_pref (p_rule 0.5, train + test) and
         round 20 p_rule 0.8 (variant S, steps 0..49) -> Round29/s4/
  check (GPU1 / CPU) data and model facts for Sec.0, flow reproduction (round 23), timings -> Round29/check/
Usage (inside Docker, from my_research/preference_inference):
  xvfb-run --auto-servernum python analyze/paper_extra/round29.py views --part i --n_parts n
  python analyze/paper_extra/round29.py encode | s1 --worker w --n_workers n | s2 | s3 | s4 | check
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

from v11core.pleasure import BIN_UNIT, BIN_ANGLES, pleasure_np, color_direction  # noqa
from v11core.rule_agent import slide_step  # noqa
from sim_v10 import layout as L  # noqa

DATA = os.path.join(_PI, 'data', 'data')
RES = os.path.join(_PI, 'data', 'result')
OUT = os.path.join(RES, 'paper_extra', 'Round29')
BOUNDS = ((-10.0, 10.0), (-10.0, 10.0))     # sim_v10/config/world/default.yml
T0, K = 23, 15
KS = np.arange(1, K + 1)
NB = 8
CONDS1 = ('p50', 'none', 'sr50', 'p80')
CONDS3 = ('p50', 'none', 'sr50')
PAIRS = tuple(range(6))
EPS = (160, 170, 180, 190, 200)
EXP = {'p50': 'v12_s1_mixed_l1_s{s}', 'none': 'v12_s1_random_l1_s{s}',
       'sr50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s{s}', 'p80': 'v13_s1_mixed_l1_s{s}'}
S1DATA = {'p50': 'v11_s1_mixed', 'sr50': 'v11_s1_mixed', 'none': 'v11_s1_random', 'p80': 'v13_s1_mixed80'}
P_RULE = {'p50': 0.5, 'sr50': 0.5, 'p80': 0.8, 'none': 0.0}
MIN_IMG, MIN_FEAT = 5, 3
REACT = 0.25


def inside(q):
    return bool(np.all(np.abs(np.asarray(q)) <= 10.0 + 1e-9))


def keff(p2, u):
    """largest k <= K such that p2 + j u is inside the arena for every j <= k (0 if none)."""
    k = 0
    for j in KS:
        if not inside(p2 + j * u):
            break
        k = j
    return k


# =============================================================== Sec.1 views (CPU, xvfb)
def views(part, n_parts, n_ep=600):
    from sim_v10 import creator
    from sim_v10.util import load_config
    cfg = load_config(os.path.join(_PI, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml'))
    env = creator.create_environment(cfg.environment)
    env.init()
    env.off_display()
    g = h5py.File(os.path.join(DATA, 'paper_r21_swap', 'data_noswap.h5'), 'r')['test']
    pos, opos, lay = g['self_position'][()], g['other_position'][()], g['layout'][()]
    idx = np.array_split(np.arange(n_ep), n_parts)[part]
    od = os.path.join(OUT, 'views')
    os.makedirs(od, exist_ok=True)

    def render(sp, op):
        env.self_agent.p = np.asarray(sp, dtype=np.float64)
        env.other_agent.p = np.asarray(op, dtype=np.float64)
        env.world.set_camera('self')
        env.world.draw(env.self_agent, env.other_agent)
        return np.round(env.world.capture() * 255).astype(np.uint8)

    nbad = 0
    with h5py.File(os.path.join(od, f'views_part{part}of{n_parts}.h5'), 'w') as f:
        f.create_dataset('idx', data=idx)
        vv = f.create_dataset('views', (len(idx), K, NB, 16, 64, 3), dtype='uint8', chunks=(1, K, NB, 16, 64, 3),
                              compression='gzip', compression_opts=4)
        vok = f.create_dataset('valid', (len(idx), K, NB), dtype='bool')
        for i, n in enumerate(idx):
            env.set_layout(L.CODE_LAYOUT[int(lay[n])])
            p2 = opos[n, 0].astype(np.float64)
            assert np.all(opos[n] == opos[n, 0])
            buf = np.zeros((K, NB, 16, 64, 3), np.uint8)
            ok = np.zeros((K, NB), bool)
            for ki, k in enumerate(KS):
                fr = T0 + k
                nbad += int(not np.array_equal(render(pos[n, fr], p2), g['self_vision'][n, fr]))
                for b in range(NB):
                    q = p2 + k * BIN_UNIT[b]
                    if inside(q):
                        ok[ki, b] = True
                        buf[ki, b] = render(pos[n, fr], q)
            vv[i], vok[i] = buf, ok
            if (i + 1) % 25 == 0:
                print(f'{i + 1}/{len(idx)}', flush=True)
        f.attrs['rerender_mismatch'] = nbad
    print('rerender mismatches', nbad, 'saved part', part, flush=True)


def load_views():
    d = os.path.join(OUT, 'views')
    parts = sorted((f for f in os.listdir(d) if f.startswith('views_part')),
                   key=lambda s: int(s[len('views_part'):].split('of')[0]))
    acc = {k: [] for k in ('idx', 'views', 'valid')}
    for f in parts:
        h = h5py.File(os.path.join(d, f), 'r')
        assert int(h.attrs['rerender_mismatch']) == 0, f
        for k in acc:
            acc[k].append(h[k][()])
    V = {k: np.concatenate(v) for k, v in acc.items()}
    assert np.array_equal(V['idx'], np.arange(len(V['idx'])))
    return V


# =============================================================== model helpers (GPU)
def torch_env():
    import torch
    import round21_eval as R21   # registers the signed class; T = eval_trial_v12, calib_common, scaled
    import util
    from p2test_v11 import read_a2
    return torch, R21, util, read_a2


def load_model(cond, s, e, dev):
    torch, R21, util, _ = torch_env()
    m, _ = R21.T.load_v12(EXP[cond].format(s=s), s, e, dev)
    m.eval()
    return m


def encode():
    torch, R21, util, _ = torch_env()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    os.makedirs(os.path.join(OUT, 'ref'), exist_ok=True)
    m = load_model('p50', 0, 200, dev)
    V = load_views()
    t0 = time.time()
    np.save(os.path.join(OUT, 'ref', 'feats_s1.npy'), enc2(m, V['views'], dev))
    np.save(os.path.join(OUT, 'ref', 'valid_s1.npy'), V['valid'])
    print(f'encode done {time.time() - t0:.0f}s', flush=True)


def _enc(module, imgs, dev, chunk=8192):
    torch, R21, util, _ = torch_env()
    sh = imgs.shape[:-3]
    flat = imgs.reshape((-1,) + imgs.shape[-3:])
    out = []
    with torch.no_grad():
        for c0 in range(0, flat.shape[0], chunk):
            x = torch.from_numpy(util.scale_vision(util.transpose_vision(flat[c0:c0 + chunk], dim=4))).to(dev)
            out.append(module(x).cpu().numpy())
    return np.concatenate(out).reshape(sh + (-1,))


def enc2(m, imgs, dev):
    return _enc(m.other_vision_encoder_module, imgs, dev)


def enc1(m, imgs, dev):
    return _enc(m.self_vision_encoder_module, imgs, dev)


# =============================================================== Sec.1 flow and read-outs
def s1_flow(model, D, m_vec, dev, batch=50, flag=0.0, Kf=K):
    """Round 23 Sec.2.2 flow with p2 = [m, 0 x 8, flag] (m_vec (2,) or None = 0). -> preds (N, Kf, 16, 64, 3), feats."""
    torch, R21, util, _ = torch_env()
    N = D['self_vision'].shape[0]
    out, fo = [], []
    with torch.no_grad():
        for b0 in range(0, N, batch):
            x = R21.scaled(D['self_vision'][b0:b0 + batch], dev)
            m = torch.from_numpy(D['self_motion'][b0:b0 + batch]).float().to(dev)
            B = x.shape[0]
            util.seed_all(0)
            model.init_state(B)
            for t in range(T0):
                model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0, 0)
            P1 = torch.zeros(B, 8, device=dev)
            Q = torch.zeros(B, 11, device=dev)
            Q[:, 10] = flag
            if m_vec is not None:
                Q[:, 0:2] = torch.tensor(np.asarray(m_vec, np.float32), device=dev)[None]
            hid = torch.ones(B, dtype=torch.bool, device=dev)
            preds, fs = [], []
            for k in range(Kf):
                pr = model({'self_vision': x[:, T0 + k], 'self_motion': m[:, T0 + k]}, 1.0, 1.0, p1_override=P1,
                           p2_override=Q, hidden_override=hid)['self_vision']
                preds.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
                fs.append(model.predict_feature(model.get_state()['other'][0]).cpu().numpy())
            out.append(np.stack(preds, 1))
            fo.append(np.stack(fs, 1))
    return np.concatenate(out), np.concatenate(fo)


def ols_slope(ks, y):
    return np.polyfit(ks, y, 1)[0]


def img_slopes(est, p2, KE):
    """est (N, K, 2); KE (N, 8) usable k range per direction -> slopes (N, 8) along BIN_UNIT[c] over k = 1..KE[n, c]
    (NaN if KE < MIN_IMG or fewer than MIN_IMG finite points)."""
    N = est.shape[0]
    S = np.full((N, NB), np.nan)
    for n in range(N):
        for c in range(NB):
            ke = KE[n, c]
            if ke < MIN_IMG:
                continue
            y = ((est[n, :ke] - p2[n]) * BIN_UNIT[c]).sum(-1)
            ok = np.isfinite(y)
            if ok.sum() >= MIN_IMG:
                S[n, c] = ols_slope(KS[:ke][ok], y[ok])
    return S


def feat_scores(f, F, valid):
    """f (N, K, 64) predicted features; F (N, K, 8, 64) reference (A-2 k steps along BIN_UNIT[b]); valid (N, K, 8).
    -> score (N, 8): mean over k (both p2 + k u_c and p2 - k u_c valid) of sign(dist to -k - dist to +k), i.e. +1 when
    closer to 'moved k steps along c' than to 'moved k steps opposite'; NaN if fewer than MIN_FEAT usable k."""
    dist = ((f[:, :, None, :] - F) ** 2).mean(-1)                            # (N, K, 8)
    S = np.full((f.shape[0], NB), np.nan)
    for c in range(NB):
        o = (c + 4) % NB
        ok = valid[:, :, c] & valid[:, :, o]
        sc = np.sign(dist[:, :, o] - dist[:, :, c])
        cnt = ok.sum(1)
        s = np.where(ok, sc, 0).sum(1) / np.maximum(cnt, 1)
        S[:, c] = np.where(cnt >= MIN_FEAT, s, np.nan)
    return S


def s1_geometry(D):
    p2 = D['other_position'][:, 0].astype(np.float64)
    KE = np.array([[keff(p, BIN_UNIT[c]) for c in range(NB)] for p in p2])
    return p2, KE


def s1_eval(cond, s, e, dev, D, F, valid, ab, n=None):
    torch, R21, util, read_a2 = torch_env()
    model = load_model(cond, s, e, dev)
    p2, KE = s1_geometry(D)
    sp = D['self_position'][:, T0:T0 + K].astype(np.float64)       # A-1 at frame T0 + k - 1 (round-21 convention)
    sav = {'KE': KE.astype(np.int8)}
    for name, mv in [('zero', None)] + [(f'd{b}', BIN_UNIT[b]) for b in range(NB)]:
        pr, fs = s1_flow(model, D, mv, dev)
        est = np.stack([read_a2(pr[:, i], sp[:, i], *ab) for i in range(K)], 1)
        sav[f'{name}_img'] = img_slopes(est, p2, KE).astype(np.float32)
        sav[f'{name}_feat'] = feat_scores(fs, F, valid).astype(np.float32)
        sav[f'{name}_est'] = est.astype(np.float32)
    return sav


def s1_jobs():
    return [(c, s, e) for c in CONDS1 for s in PAIRS for e in EPS]


def s1_main(worker, n_workers):
    torch, R21, util, read_a2 = torch_env()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    ab = R21.calib_common()
    D = R21.load_data('noswap')
    F = np.load(os.path.join(OUT, 'ref', 'feats_s1.npy'))
    valid = np.load(os.path.join(OUT, 'ref', 'valid_s1.npy'))
    od = os.path.join(OUT, 's1')
    os.makedirs(od, exist_ok=True)
    for i, (c, s, e) in enumerate(s1_jobs()):
        if i % n_workers != worker:
            continue
        f = os.path.join(od, f'{c}_s{s}_ep{e}.npz')
        if os.path.exists(f):
            continue
        t0 = time.time()
        sav = s1_eval(c, s, e, dev, D, F, valid, ab)
        np.savez_compressed(f + '.tmp.npz', **sav)
        os.replace(f + '.tmp.npz', f)
        torch.cuda.empty_cache()
        print(f's1:{c}:{s}:{e} {time.time() - t0:.0f}s {time.strftime("%m-%d %H:%M")}', flush=True)


# =============================================================== Sec.2 (CPU)
def sector(m):
    """m (..., 2) -> sector 0..7 centred on BIN_ANGLES (range b's direction); -1 where |m| < 0.5 (asleep)."""
    th = np.arctan2(m[..., 1], m[..., 0])
    k = np.round((th - BIN_ANGLES[0]) / (np.pi / 4)).astype(int) % NB
    return np.where(np.linalg.norm(m, axis=-1) < 0.5, -1, k)


def pleasure_all(sv, chunk=200):
    """sv (N, T, 16, 64, 3) uint8 -> (N, T, 8) float32 red - green (scale 1, as compute_pleasure)."""
    out = []
    for c0 in range(0, sv.shape[0], chunk):
        out.append(pleasure_np(sv[c0:c0 + chunk]))
    return np.concatenate(out).astype(np.float32)


def signs(cond, split, n):
    if cond != 'sr50':
        return np.ones(n, np.float32)
    lk = h5py.File(os.path.join(DATA, 'paper_s1_mixed50_signrand', 'signs.h5'), 'r')[split]['a1_liked'][()]
    return (2.0 * lk - 1.0).astype(np.float32)


def rule_dir(v, p, liked):
    """the rule step of rule_agent.AgentPolicy (None if the liked colour is not visible or no slide is left)."""
    d = color_direction(v, liked)
    if d is None:
        return None
    return slide_step(p, d, BOUNDS, 1.0)


def s2_data(cond):
    g = h5py.File(os.path.join(DATA, S1DATA[cond], 'data.h5'), 'r')
    out = {}
    for split in ('train', 'test'):
        h = g[split]
        sv = h['self_vision'][()]
        P = pleasure_all(sv) * signs(cond, split, sv.shape[0])[:, None, None]
        sec = sector(h['self_motion'][()].astype(np.float64))
        d = {'P': P, 'sec': sec, 'src': h['a1_source'][()]}
        if split == 'test':
            pos = h['self_position'][()].astype(np.float64)
            best = np.full(sec.shape, -1)
            rule_ok = np.zeros(sec.shape, bool)
            for n in range(sv.shape[0]):
                for t in range(sv.shape[1]):
                    u = rule_dir(sv[n, t], pos[n, t], 'Red')
                    if u is not None:
                        best[n, t] = sector(u)
                        rule_ok[n, t] = True
            d['best_sec'], d['rule_ok'] = best, rule_ok
        out[split] = d
        del sv
    return out


def s2_train(X, y, seed, kind, epochs=20, batch=1024, lr=1e-3):
    import torch
    torch.manual_seed(seed)
    net = (torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8)) if kind == 'mlp'
           else torch.nn.Linear(8, 8))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    g = torch.Generator().manual_seed(seed)
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(len(yt), generator=g)
        tot = 0.0
        for b0 in range(0, len(yt), batch):
            ib = perm[b0:b0 + batch]
            loss = torch.nn.functional.cross_entropy(net(Xt[ib]), yt[ib])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(ib)
        losses.append(tot / len(yt))
    return net, losses


def s2_main():
    import torch
    torch.set_num_threads(2)
    od = os.path.join(OUT, 's2')
    os.makedirs(od, exist_ok=True)
    summ = {}
    for cond in ('p50', 'p80', 'none', 'sr50'):
        t0 = time.time()
        D = s2_data(cond)
        tr, te = D['train'], D['test']
        mtr = tr['sec'] >= 0
        X, y = tr['P'][mtr], tr['sec'][mtr]
        freq = np.bincount(y, minlength=NB) / len(y)
        fsec = int(np.argmax(freq))
        awake = te['sec'] >= 0
        Ne = te['sec'].shape[0]

        def ep_acc(pred):
            c = (pred == te['sec']) & awake
            return c.sum(1) / np.maximum(awake.sum(1), 1)

        best_pred = np.where(te['rule_ok'], te['best_sec'], fsec) if P_RULE[cond] > 0 else np.full(te['sec'].shape, fsec)
        lev = {'chance': np.full(Ne, 1.0 / NB), 'freq': ep_acc(np.full(te['sec'].shape, fsec)), 'best': ep_acc(best_pred)}
        # oracle check: rule steps (source 1) of the test split equal the oracle rule step exactly
        r1 = (te['src'] == 1)
        chk = {'n_rule_steps': int(r1.sum()), 'rule_ok_on_rule_steps': float(te['rule_ok'][r1].mean()) if r1.any() else None}
        sav = {k: v.astype(np.float32) for k, v in lev.items()}
        react = {}
        for kind in ('mlp', 'lin'):
            for s in PAIRS:
                net, losses = s2_train(X, y, s, kind)
                with torch.no_grad():
                    logits = net(torch.from_numpy(te['P'].reshape(-1, 8))).numpy().reshape(te['sec'].shape + (NB,))
                    E = np.zeros((2 * NB, 8), np.float32)
                    for b in range(NB):
                        E[b, b] = REACT
                        E[NB + b, b] = -REACT
                    pr = torch.softmax(net(torch.from_numpy(E)), -1).numpy()
                sav[f'{kind}_s{s}_acc'] = ep_acc(logits.argmax(-1)).astype(np.float32)
                sav[f'{kind}_s{s}_react'] = pr.astype(np.float32)
                react[f'{kind}_s{s}_loss'] = losses
        np.savez_compressed(os.path.join(od, f'{cond}.npz'), **sav)
        summ[cond] = {'n_train_steps': int(len(y)), 'freq': freq.tolist(), 'freq_sector': fsec,
                      'level_means': {k: float(v.mean()) for k, v in lev.items()}, 'oracle_check': chk,
                      'rule_ok_frac_test_awake': float(te['rule_ok'][awake].mean()),
                      'train_loss': react, 'sec': time.time() - t0}
        print(cond, json.dumps(summ[cond]['level_means']), f'{time.time() - t0:.0f}s', flush=True)
    json.dump(summ, open(os.path.join(od, 's2_run.json'), 'w'), indent=1)


# =============================================================== Sec.3 (GPU)
def ridge_fit(X, Y, alpha=1.0):
    Xm, Ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - Xm, Y - Ym
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return W, Ym - Xm @ W


def show_flow(model, sv, sm, dev, batch=50):
    """all frames shown (p_mask 0), process-1 motion visible (motion_masking off, as eval_inverse_v14.load);
    -> predict_feature(os) at steps 0..T-2 (prediction of Encoder-2 of frame t + 1), (N, T-1, 64)."""
    torch, R21, util, _ = torch_env()
    model.motion_masking = False
    out = []
    with torch.no_grad():
        util.seed_all(0)
        for b0 in range(0, sv.shape[0], batch):
            x = R21.scaled(sv[b0:b0 + batch], dev)
            m = torch.from_numpy(sm[b0:b0 + batch]).float().to(dev)
            model.init_state(x.shape[0])
            fs = []
            for t in range(x.shape[1] - 1):
                model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0, 0)
                fs.append(model.predict_feature(model.get_state()['other'][0]).cpu().numpy())
            out.append(np.stack(fs, 1))
    return np.concatenate(out)


def freeze_feats(model, sv, sm, dev, batch=50, Kf=10):
    """round 23 freeze (frames 0..22 shown, freeze from 23, p1 = 0, motion hidden) with p2 = 0 (stage-I input);
    -> predict_feature(os) at steps 23..32 (predictions of frames 24..33)."""
    torch, R21, util, _ = torch_env()
    D = {'self_vision': sv, 'self_motion': sm}
    model.motion_masking = True
    return s1_flow(model, D, None, dev, batch=batch, flag=0.0, Kf=Kf)[1]


def suff(pred, targ, bins):
    """per-episode sufficient statistics over (t, bins) -> (N, 6): n, Sx, Sy, Sxx, Syy, Sxy and (N,) SSE."""
    x = pred[..., bins].reshape(pred.shape[0], -1).astype(np.float64)
    y = targ[..., bins].reshape(targ.shape[0], -1).astype(np.float64)
    st = np.stack([np.full(x.shape[0], x.shape[1], np.float64), x.sum(1), y.sum(1), (x * x).sum(1), (y * y).sum(1),
                   (x * y).sum(1), ((y - x) ** 2).sum(1)], 1)
    return st


def s3_main():
    torch, R21, util, _ = torch_env()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    od = os.path.join(OUT, 's3')
    os.makedirs(od, exist_ok=True)
    BINSETS = {'c0167': [0, 1, 6, 7], 'c2345': [2, 3, 4, 5], **{f'b{b}': [b] for b in range(NB)}}
    run = {}
    cache = {}
    for cond in ('p50', 'sr50', 'none'):      # p50 and sr50 share the data (loaded once)
        ds = S1DATA[cond]
        if ds not in cache:
            g = h5py.File(os.path.join(DATA, ds, 'data.h5'), 'r')
            cache = {ds: {sp: {k: g[sp][k][()] for k in ('self_vision', 'other_vision', 'self_motion')}
                          for sp in ('train', 'test')}}
        G = cache[ds]
        Pa1_tr = pleasure_all(G['train']['self_vision'])
        Pa1_te = pleasure_all(G['test']['self_vision'])
        Pa2_te = pleasure_all(G['test']['other_vision'])
        for s in PAIRS:
            t0 = time.time()
            m = load_model(cond, s, 200, dev)
            Xtr = enc1(m, G['train']['self_vision'], dev)
            W, b = ridge_fit(Xtr.reshape(-1, Xtr.shape[-1]).astype(np.float64), Pa1_tr.reshape(-1, 8).astype(np.float64))
            rd = lambda Z: (Z.astype(np.float64) @ W + b)  # noqa
            rows = {
                'own': (rd(enc1(m, G['test']['self_vision'], dev)), Pa1_te),
                'other_real': (rd(enc1(m, G['test']['other_vision'], dev)), Pa2_te),
                'enc2_real': (rd(enc2(m, G['test']['self_vision'], dev)), Pa2_te),
                'pred_show': (rd(show_flow(m, G['test']['self_vision'], G['test']['self_motion'], dev)), Pa2_te[:, 1:]),
                'pred_freeze': (rd(freeze_feats(load_model(cond, s, 200, dev), G['test']['self_vision'],
                                                G['test']['self_motion'], dev)), Pa2_te[:, T0 + 1:T0 + 11]),
            }
            sav = {'W': W.astype(np.float32), 'b': b.astype(np.float32)}
            for rn, (pr, tg) in rows.items():
                for bn, bins in BINSETS.items():
                    sav[f'{rn}|{bn}'] = suff(pr, tg, bins)
            np.savez_compressed(os.path.join(od, f'{cond}_s{s}.npz'), **sav)
            run[f'{cond}_s{s}'] = {'sec': time.time() - t0, 'enc1_sum': float(np.abs(Xtr).sum())}
            print(f's3:{cond}:{s} {time.time() - t0:.0f}s', flush=True)
            del Xtr
    json.dump(run, open(os.path.join(od, 's3_run.json'), 'w'), indent=1)


# =============================================================== Sec.4 (CPU)
def angle_bin(m, nb):
    th = np.arctan2(m[..., 1], m[..., 0])
    return (np.floor((th + np.pi) / (2 * np.pi / nb)).astype(int)) % nb


def observer(ov, pos, mot, src, lay, liked_true, p, nsteps, nbins_list=(72, 8)):
    """per-step log LR log P(m | true) - log P(m | false) (natural log) for each direction resolution, plus
    the 'arrived' flag (A-2 within 1.5 of its liked colour's reference point at or before that step)."""
    N = ov.shape[0]
    out = {nb: np.zeros((N, nsteps)) for nb in nbins_list}
    arrived = np.zeros((N, nsteps), bool)
    match_true = np.zeros((N, nsteps), bool)
    info_ok = np.zeros((N, nsteps), bool)
    for n in range(N):
        ltrue = 'Red' if liked_true[n] == 1 else 'Green'
        lfalse = 'Green' if ltrue == 'Red' else 'Red'
        ref = L.LANDMARKS[L.CODE_LAYOUT[int(lay[n])]][ltrue]
        arr = False
        for t in range(nsteps):
            arr = arr or np.linalg.norm(pos[n, t] - ref) <= 1.5
            arrived[n, t] = arr
            m = mot[n, t].astype(np.float64)
            if np.linalg.norm(m) < 0.5:          # asleep: no information
                continue
            uT = rule_dir(ov[n, t], pos[n, t], ltrue)
            uF = rule_dir(ov[n, t], pos[n, t], lfalse)
            if src[n, t] == 1:
                match_true[n, t] = uT is not None and np.allclose(m, uT, atol=1e-5)
            info_ok[n, t] = True
            for nb in nbins_list:
                bm = angle_bin(m, nb)

                def lik(u):
                    if u is None:
                        return 1.0 / nb
                    return p * float(angle_bin(u, nb) == bm) + (1 - p) / nb
                out[nb][n, t] = np.log(lik(uT)) - np.log(lik(uF))
    return out, arrived, match_true, info_ok


def s4_main():
    od = os.path.join(OUT, 's4')
    os.makedirs(od, exist_ok=True)
    run = {}
    srcs = [('p05', os.path.join(DATA, 'v11_s2_pref', 'data.h5'), ('train', 'test'), 0.5, 101),
            ('p08', os.path.join(DATA, 'paper_r20_swap_p80', 'data_S.h5'), ('test',), 0.8, 50)]
    for name, path, splits, p, nst in srcs:
        t0 = time.time()
        acc = {}
        g = h5py.File(path, 'r')
        for split in splits:
            h = g[split]
            ov, pos, mot = h['other_vision'][:, :nst], h['other_position'][:, :nst], h['other_motion'][:, :nst]
            res, arr, mt, ok = observer(ov, pos, mot, h['a2_source'][:, :nst], h['layout'][()], h['a2_liked'][()], p, nst)
            for nb, v in res.items():
                acc.setdefault(f'llr{nb}', []).append(v)
            for k, v in (('arrived', arr), ('match_true', mt), ('info_ok', ok), ('src', h['a2_source'][:, :nst]),
                         ('liked', np.repeat(h['a2_liked'][()][:, None], 1, 1))):
                acc.setdefault(k, []).append(v)
        sav = {k: np.concatenate(v) for k, v in acc.items()}
        np.savez_compressed(os.path.join(od, f'{name}.npz'), **sav)
        r1 = sav['src'] == 1
        run[name] = {'n_episodes': int(sav['llr72'].shape[0]), 'n_steps': nst, 'p_rule': p,
                     'rule_steps': int(r1.sum()), 'rule_step_equals_true_rule_frac': float(sav['match_true'][r1].mean()),
                     'sec': time.time() - t0}
        print(name, json.dumps(run[name]), flush=True)
    json.dump(run, open(os.path.join(od, 's4_run.json'), 'w'), indent=1)


# =============================================================== check
def check():
    torch, R21, util, read_a2 = torch_env()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    os.makedirs(os.path.join(OUT, 'check'), exist_ok=True)
    out = {}
    # (a) data facts for Sec.0
    facts = {}
    for name, path, split in (('v11_s1_random', 'v11_s1_random/data.h5', 'test'), ('v11_s1_mixed', 'v11_s1_mixed/data.h5', 'test'),
                              ('v13_s1_mixed80', 'v13_s1_mixed80/data.h5', 'test'), ('v11_s2_pref', 'v11_s2_pref/data.h5', 'test'),
                              ('r20_p80_S', 'paper_r20_swap_p80/data_S.h5', 'test'), ('r21_noswap', 'paper_r21_swap/data_noswap.h5', 'test')):
        h = h5py.File(os.path.join(DATA, path), 'r')[split]
        sm, om = h['self_motion'][()], h['other_motion'][()]
        ns, no = np.linalg.norm(sm, axis=-1), np.linalg.norm(om, axis=-1)
        fx = {'shape': list(h['self_vision'].shape[:2]),
              'self_motion_norm_frac_unit': float((np.abs(ns - 1) < 1e-4).mean()), 'self_motion_frac_zero': float((ns == 0).mean()),
              'other_motion_norm_frac_unit': float((np.abs(no - 1) < 1e-4).mean()), 'other_motion_frac_zero': float((no == 0).mean()),
              'self_pos_absmax': float(np.abs(h['self_position'][()]).max()), 'other_pos_absmax': float(np.abs(h['other_position'][()]).max())}
        if 'a1_source' in h:
            fx['a1_source_counts'] = np.bincount(h['a1_source'][()].ravel(), minlength=4).tolist()
        if 'a2_source' in h:
            fx['a2_source_counts'] = np.bincount(h['a2_source'][()].ravel(), minlength=4).tolist()
            fx['a2_liked_counts'] = np.bincount(h['a2_liked'][()], minlength=2).tolist()
        facts[name] = fx
    vs = {v: h5py.File(os.path.join(DATA, 'paper_r20_swap_p80', f'data_{v}.h5'), 'r')['test'] for v in 'NOSR'}
    facts['r20_p80_first50_identical_NOSR'] = {k: all(np.array_equal(vs['S'][k][:, :50], vs[v][k][:, :50]) for v in 'NOR')
                                               for k in ('other_motion', 'other_position', 'a2_source')}
    sg = h5py.File(os.path.join(DATA, 'paper_s1_mixed50_signrand', 'signs.h5'), 'r')
    facts['signrand50_kept_frac'] = {sp: float(sg[sp]['a1_liked'][()].mean()) for sp in ('train', 'test')}
    out['data_facts'] = facts
    # (b) model facts: encoders identical, one shared LSTMCell, process-2 input size
    ref = load_model('p50', 0, 200, dev).state_dict()
    same = {}
    for c in CONDS1:
        for s in PAIRS:
            sd = load_model(c, s, 200, dev).state_dict()
            for mod in ('self_vision_encoder_module', 'other_vision_encoder_module'):
                ks = [k for k in ref if k.startswith(mod)]
                same.setdefault(mod, []).append(all(torch.equal(ref[k], sd[k]) for k in ks))
    m = load_model('p50', 0, 200, dev)
    out['model_facts'] = {'encoders_identical_all_24': {k: bool(all(v)) for k, v in same.items()},
                          'superposition_params': {k: list(v.shape) for k, v in m.superposition_module.state_dict().items()},
                          'n_lstm_cells_in_superposition': sum(1 for x in m.superposition_module.modules()
                                                               if isinstance(x, torch.nn.LSTMCell)),
                          'enc1_dim': int(enc1(m, np.zeros((1, 16, 64, 3), np.uint8), dev).shape[-1])}
    # (c) flow reproduction: our s1_flow with round 23's zero input (flag 1, K = 10) equals round 23's stored zero_est
    n = 50
    D = R21.load_data('noswap', n)
    ab = R21.calib_common()
    pr, _ = s1_flow(m, D, None, dev, flag=1.0, Kf=10)
    sp = D['self_position'][:, 23:33].astype(np.float64)
    est = np.stack([read_a2(pr[:, i], sp[:, i], *ab) for i in range(10)], 1).astype(np.float32)
    z23 = np.load(os.path.join(RES, 'paper_extra', 'Round23', 'eval_dir', 'p50_s0_ep200.npz'))['zero_est'][:n]
    out['flow_repro_vs_round23_zero_est'] = {'max_abs_diff': float(np.nanmax(np.abs(est - z23))),
                                             'nan_same': bool(np.array_equal(np.isnan(est), np.isnan(z23)))}
    # (d) Sec.1 timing on 50 episodes (values not looked at), shapes only
    if os.path.exists(os.path.join(OUT, 'ref', 'feats_s1.npy')):
        F = np.load(os.path.join(OUT, 'ref', 'feats_s1.npy'), mmap_mode='r')[:n]
        valid = np.load(os.path.join(OUT, 'ref', 'valid_s1.npy'))[:n]
        t0 = time.time()
        sav = s1_eval('p50', 0, 200, dev, D, np.asarray(F), valid, ab)
        out['s1_timing_50ep_s'] = time.time() - t0
        out['s1_keys'] = {k: list(v.shape) for k, v in sav.items()}
        out['s1_finite_frac'] = {k: float(np.isfinite(v).mean()) for k, v in sav.items() if k.endswith(('_img', '_feat'))}
        V = np.load(os.path.join(OUT, 'ref', 'valid_s1.npy'))
        p2, KE = s1_geometry(R21.load_data('noswap'))
        out['s1_geometry'] = {'KE_ge5_frac': float((KE >= MIN_IMG).mean()), 'KE_mean': float(KE.mean()),
                              'feat_usable_k_ge3_frac': float(np.mean([[(V[n2, :, c] & V[n2, :, (c + 4) % 8]).sum() >= MIN_FEAT
                                                                      for c in range(8)] for n2 in range(V.shape[0])])),
                              'valid_frac': float(V.mean())}
    json.dump(out, open(os.path.join(OUT, 'check', 'check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('views', 'encode', 's1', 's2', 's3', 's4', 'check'))
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--n_parts', type=int, default=1)
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--n_workers', type=int, default=1)
    a = ap.parse_args()
    if a.mode == 'views':
        views(a.part, a.n_parts)
    elif a.mode == 'encode':
        encode()
    elif a.mode == 's1':
        s1_main(a.worker, a.n_workers)
    elif a.mode == 's2':
        s2_main()
    elif a.mode == 's3':
        s3_main()
    elif a.mode == 's4':
        s4_main()
    else:
        check()
