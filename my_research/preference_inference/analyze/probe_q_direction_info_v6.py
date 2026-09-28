"""
(c) 8-direction probe-Q direction-information check
(docs/v6_experiment_log.md §19, pre-registered before measuring).

For a critic and a frame set, computes the probe-Q matrix Q_ik = Q(v_i, a_k
[, r]) over the same K=8 unit-vector directions the SM input uses
(model/model.py SuperpositionNetworkProbeQ), then

    D = mean_i Var_k(Q_ik)            direction-to-direction spread within a frame
    S = Var_i( mean_k Q_ik )          frame-to-frame spread of the direction-mean Q
    R = D / S

and the decision statistic rho = R(candidate) / R(reference) on identical
frames. Reference = v3_rl_critic.pth (the critic the v4/v5 configs load).

Conditions (as used in the real experiments):
  c1: r = A-1 (+1,-1,0,0), r3_stay test split, self_vision   (base process-1)
  c2: r = A-2 (-1,+1,0,0), r2_a1random_a2rl test split, other_vision
      (oracle process-2)

Informational extras: bootstrap CI of rho over episodes, the tanh-normalised
ratio, and whether the direction pattern points toward the preferred landmark.

Usage (inside Docker):
    cd /work/my_research/preference_inference
    python3 analyze/probe_q_direction_info_v6.py
"""
import json
import math
import os
import sys

sys.path.insert(0, '/work')

import h5py
import numpy as np
import torch

from my_research.rl_agent_sac import CriticLSTM as CriticV3
from my_research.preference_inference.model.rl_agent_sac_v6 import (
    CriticLSTM as CriticV6, A1_TRUE_R, A2_TRUE_R,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = '/work/my_research/preference_inference/data/model'
DATA_DIR = '/work/my_research/preference_inference/data/data'
SAVE_DIR = '/work/my_research/preference_inference/data/result/v6_baseline'

K = 8
PROBES = np.array([[math.cos(2 * math.pi * i / K), math.sin(2 * math.pi * i / K)]
                   for i in range(K)], dtype=np.float32)

CONDITIONS = {
    'c1': dict(dataset='r3_stay', vision='self_vision', position='self_position',
               r=A1_TRUE_R, landmark=np.array([-9.0, 9.0]), landmark_name='Red'),
    'c2': dict(dataset='r2_a1random_a2rl', vision='other_vision', position='other_position',
               r=A2_TRUE_R, landmark=np.array([-9.0, -9.0]), landmark_name='Green'),
    # 2x2 deconfound (2026-09-28, user request, docs/v6_experiment_log.md §23):
    # c1/c2 differ in BOTH dataset and r simultaneously. c3/c4 swap only r within
    # each row's existing vision/position source, to separate "r is hard" from
    # "this dataset's vision/position distribution is hard".
    'c3': dict(dataset='r3_stay', vision='self_vision', position='self_position',
               r=A2_TRUE_R, landmark=np.array([-9.0, -9.0]), landmark_name='Green'),
    'c4': dict(dataset='r2_a1random_a2rl', vision='other_vision', position='other_position',
               r=A1_TRUE_R, landmark=np.array([-9.0, 9.0]), landmark_name='Red'),
}

# name, kind ('v3' no r | 'ctrl' r_dim=0 | 'v6' r-conditioned), file, film, conditions
CRITICS = [
    ('v3_A1_s0_REF', 'v3', 'v3_rl_critic.pth', False, ['c1', 'c2', 'c3', 'c4']),
    ('v3_A1_s1', 'v3', 'v3_rl_critic_seed1.pth', False, ['c1', 'c2']),
    ('v3_A1_s2(collapsed)', 'v3', 'v3_rl_critic_seed2.pth', False, ['c1', 'c2']),
    ('v3_A2_s0', 'v3', 'v3_rl_a2_critic.pth', False, ['c2']),
    ('ctrl_s0', 'ctrl', 'v6_rl_critic_seed0_control.pth', False, ['c1']),
    ('ctrl_s1', 'ctrl', 'v6_rl_critic_seed1_control.pth', False, ['c1']),
    ('ctrl_s2', 'ctrl', 'v6_rl_critic_seed2_control.pth', False, ['c1']),
    ('film_s0', 'v6', 'v6_rl_critic_seed0_film.pth', True, ['c1', 'c2']),
    ('film_s1', 'v6', 'v6_rl_critic_seed1_film.pth', True, ['c1', 'c2']),
    ('film_s2', 'v6', 'v6_rl_critic_seed2_film.pth', True, ['c1', 'c2']),
    ('trial1_s0(r,noRelabel)', 'v6', 'v6_rl_critic_seed0.pth', False, ['c1', 'c2']),
    ('relabel_s0(trial2)', 'v6', 'v6_rl_critic_seed0_relabel.pth', False, ['c1', 'c2']),
    ('relabel_s2', 'v6', 'v6_rl_critic_seed2_relabel.pth', False, ['c1', 'c2']),
    ('norelabel_s1', 'v6', 'v6_rl_critic_seed1_norelabel.pth', False, ['c1', 'c2']),
    # added after the first run, informational only: state-collapsed critics (Q constant
    # across states, so S~0 and R/rho are meaningless; D is still shown for completeness)
    ('relabel_s1(state-collapsed)', 'v6', 'v6_rl_critic_seed1_relabel.pth', False, ['c1', 'c2']),
    ('norelabel_s2(state-collapsed)', 'v6', 'v6_rl_critic_seed2_norelabel.pth', False, ['c1', 'c2']),
    # 2026-09-28, (a'), §21: FiLM + no-relabel
    ('film_norelabel_s0', 'v6', 'v6_rl_critic_seed0_film_norelabel.pth', True, ['c1', 'c2', 'c3', 'c4']),
    ('film_norelabel_s1(state-collapsed)', 'v6', 'v6_rl_critic_seed1_film_norelabel.pth', True, ['c1', 'c2', 'c3', 'c4']),
    ('film_norelabel_s2', 'v6', 'v6_rl_critic_seed2_film_norelabel.pth', True, ['c1', 'c2', 'c3', 'c4']),
]
REF = 'v3_A1_s0_REF'
CANDIDATES = ['film_norelabel_s0', 'film_norelabel_s2']  # s1 excluded: state-collapsed
STATE_COLLAPSE_EPS = 1e-6  # S below this -> rho is degenerate/meaningless, excluded from verdict
N_BOOT = 1000
BOOT_SEED = 0


def load_critic(kind, fname, film):
    if kind == 'v3':
        c = CriticV3()
    elif kind == 'ctrl':
        c = CriticV6(r_dim=0)
    else:
        c = CriticV6(r_dim=4, film=film)
    c.load_state_dict(torch.load(os.path.join(MODEL_DIR, fname), map_location='cpu'), strict=True)
    return c.to(DEVICE).eval()


def load_condition(cond):
    with h5py.File(os.path.join(DATA_DIR, cond['dataset'], 'data.h5'), 'r') as f:
        vis = f['test'][cond['vision']][:]        # (E,T,16,64,3) uint8
        pos = f['test'][cond['position']][:]      # (E,T,2)
    e, t = vis.shape[:2]
    frames = torch.from_numpy(vis.reshape(e * t, *vis.shape[2:]))
    return frames, pos.reshape(e * t, 2).astype(np.float64), e, t


@torch.no_grad()
def probe_q(critic, kind, frames, r_vec, batch=512):
    out = []
    probes = torch.from_numpy(PROBES).to(DEVICE)
    for i in range(0, frames.shape[0], batch):
        v = frames[i:i + batch].to(DEVICE).float().div(255.0).permute(0, 3, 1, 2).contiguous()
        b = v.shape[0]
        v_rep = v.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(b * K, *v.shape[1:])
        a_rep = probes.unsqueeze(0).expand(b, -1, -1).reshape(b * K, 2)
        if kind == 'v3':
            q, _ = critic(v_rep, a_rep, hidden=None)
        else:
            r = torch.zeros(b * K, 0, device=DEVICE) if kind == 'ctrl' else \
                torch.from_numpy(np.asarray(r_vec, dtype=np.float32)).to(DEVICE).unsqueeze(0).expand(b * K, -1)
            q, _ = critic(v_rep, a_rep, r, hidden=None)
        out.append(q.reshape(b, K).cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def ratio_stats(q):
    var_k = q.var(axis=1)
    mean_k = q.mean(axis=1)
    return var_k, mean_k, var_k.mean(), mean_k.var()


def direction_meaning(q, pos, landmark):
    d = landmark[None, :] - pos
    dist = np.linalg.norm(d, axis=1)
    ok = dist > 1e-6
    dn = d[ok] / dist[ok, None]
    cosd = dn @ PROBES.T.astype(np.float64)               # (n,K)
    qc = q[ok] - q[ok].mean(axis=1, keepdims=True)
    cc = cosd - cosd.mean(axis=1, keepdims=True)
    num = (qc * cc).sum(axis=1)
    den = np.sqrt((qc ** 2).sum(axis=1) * (cc ** 2).sum(axis=1))
    valid = den > 1e-12
    corr = float(np.mean(num[valid] / den[valid]))
    within45 = float(np.mean(cosd[np.arange(cosd.shape[0]), q[ok].argmax(axis=1)] >= math.cos(math.pi / 4)))
    return corr, within45


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    results = {}
    agg = {}   # (critic, cond) -> per-episode sums for bootstrap (raw and tanh)
    for cname, cond in CONDITIONS.items():
        frames, pos, e, t = load_condition(cond)
        print(f'[{cname}] {cond["dataset"]}/{cond["vision"]}: {e} episodes x {t} steps = {frames.shape[0]} frames', flush=True)
        for name, kind, fname, film, conds in CRITICS:
            if cname not in conds:
                continue
            critic = load_critic(kind, fname, film)
            q = probe_q(critic, kind, frames, cond['r'])
            var_k, mean_k, D, S = ratio_stats(q)
            R = D / S
            mu, sd = q.mean(), q.std()
            qt = np.tanh((q - mu) / sd)
            vt, mt, Dt, St = ratio_stats(qt)
            corr, w45 = direction_meaning(q, pos, cond['landmark'])
            results[(name, cname)] = dict(D=D, S=S, R=R, R_tanh=Dt / St, dir_corr=corr, within45=w45,
                                          q_mean=float(q.mean()), q_std=float(q.std()))
            agg[(name, cname)] = dict(
                raw=[a.reshape(e, t).sum(1) for a in (var_k, mean_k, mean_k ** 2)],
                tanh=[a.reshape(e, t).sum(1) for a in (vt, mt, mt ** 2)], n=t)
            print(f'   {name:24s} D={D:.5f} S={S:.5f} R={R:.5f} Rtanh={Dt / St:.5f} '
                  f'dirCorr={corr:+.3f} within45={w45:.3f}', flush=True)

    rng = np.random.RandomState(BOOT_SEED)
    e = agg[(REF, 'c1')]['raw'][0].shape[0]
    counts = [np.bincount(rng.randint(0, e, e), minlength=e) for _ in range(N_BOOT)]

    def boot_R(a, key):
        sv, sm, sm2 = a[key]
        t = a['n']
        out = []
        for c in counts:
            n = c.sum() * t
            D = (c @ sv) / n
            m1 = (c @ sm) / n
            S = (c @ sm2) / n - m1 ** 2
            out.append(D / S)
        return np.array(out)

    rho = {}
    for cname in CONDITIONS:
        if (REF, cname) not in results:
            print(f'[skip rho for {cname}: reference {REF} was not evaluated on this condition]')
            continue
        ref = results[(REF, cname)]
        ref_b = boot_R(agg[(REF, cname)], 'raw')
        ref_bt = boot_R(agg[(REF, cname)], 'tanh')
        for name, kind, fname, film, conds in CRITICS:
            if cname not in conds:
                continue
            r_ = results[(name, cname)]
            b = boot_R(agg[(name, cname)], 'raw') / ref_b
            bt = boot_R(agg[(name, cname)], 'tanh') / ref_bt
            rho[(name, cname)] = dict(
                rho=r_['R'] / ref['R'], rho_ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                rho_tanh=r_['R_tanh'] / ref['R_tanh'],
                rho_tanh_ci=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))])

    lines = []
    for cname, cond in CONDITIONS.items():
        lines.append(f'=== {cname}: {cond["dataset"]} {cond["vision"]}, r={cond["r"].tolist()}, '
                     f'preferred landmark {cond["landmark_name"]} ===')
        lines.append(f'{"critic":24s} {"D":>9s} {"S":>9s} {"R":>9s} {"rho":>8s} {"rho 95% CI":>17s} '
                     f'{"rho_tanh":>9s} {"dirCorr":>8s} {"in45":>6s}')
        for name, kind, fname, film, conds in CRITICS:
            if cname not in conds:
                continue
            r_, p_ = results[(name, cname)], rho[(name, cname)]
            lines.append(f'{name:24s} {r_["D"]:9.5f} {r_["S"]:9.5f} {r_["R"]:9.5f} {p_["rho"]:8.3f} '
                         f'[{p_["rho_ci"][0]:6.3f},{p_["rho_ci"][1]:6.3f}] {p_["rho_tanh"]:9.3f} '
                         f'{r_["dir_corr"]:+8.3f} {r_["within45"]:6.3f}')
        lines.append('')

    # pre-registered verdict (§19.4)
    lines.append('=== pre-registered verdict (§19.4): best seed of min over conditions of rho ===')
    scores = {}
    excluded = []
    for name in CANDIDATES:
        s_c1 = results[(name, 'c1')]['S']
        s_c2 = results[(name, 'c2')]['S']
        s_ref1 = results[(REF, 'c1')]['S']
        s_ref2 = results[(REF, 'c2')]['S']
        if min(s_c1, s_c2, s_ref1, s_ref2) < STATE_COLLAPSE_EPS:
            excluded.append(name)
            lines.append(f'  {name}: EXCLUDED (state-collapsed, S_c1={s_c1:.2e} S_c2={s_c2:.2e} -- '
                         f'rho would be degenerate/meaningless)')
            continue
        scores[name] = min(rho[(name, 'c1')]['rho'], rho[(name, 'c2')]['rho'])
        lines.append(f'  {name}: rho_c1={rho[(name, "c1")]["rho"]:.3f} rho_c2={rho[(name, "c2")]["rho"]:.3f} '
                     f'-> score(min)={scores[name]:.3f}')
    if not scores:
        lines.append('  ALL CANDIDATES EXCLUDED (state-collapsed) -- no verdict possible, inconclusive by default')
        with open(os.path.join(SAVE_DIR, 'probe_dirinfo_v6.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('\n'.join(lines))
        return
    best_name = max(scores, key=scores.get)
    best = scores[best_name]
    row = 'row1 (>=1/2): not fatal downstream -> proceed to S2' if best >= 0.5 else \
        ('row2 (<1/5): direction information lost -> non-stationarity countermeasures (a)' if best < 0.2
         else 'row3 (1/5..1/2): hard to judge -> consult on the criterion')
    lines.append(f'  best = {best_name} score={best:.3f} -> {row}')
    txt = '\n'.join(lines)
    print('\n' + txt)

    with open(os.path.join(SAVE_DIR, 'probe_dirinfo_v6.txt'), 'w') as f:
        f.write(txt + '\n')
    with open(os.path.join(SAVE_DIR, 'probe_dirinfo_v6.json'), 'w') as f:
        json.dump({'results': {f'{k[0]}|{k[1]}': v for k, v in results.items()},
                   'rho': {f'{k[0]}|{k[1]}': v for k, v in rho.items()},
                   'scores': scores, 'best': best_name, 'best_score': best, 'verdict': row,
                   'n_boot': N_BOOT, 'boot_seed': BOOT_SEED}, f, indent=2)
    print('Saved probe_dirinfo_v6.{txt,json}')


if __name__ == '__main__':
    main()
