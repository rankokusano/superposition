"""
Feasibility check for a goal set where the state matters
(docs/v7_experiment_log.md §2.10). No retraining; no v7 dataset is used.

  --rollout   seed1 actor drives A-2 exactly as in collect_data_v7.py (other
              camera, A-1 RandomAgent, A-2 clipped to +-10, stochastic
              sample(), hidden carried), 200 episodes x 101 frames per goal.
              Saves A-2 position / action / vision per goal (npz, not in git).
  --check1    actor: where does each goal's episode end, by start position.
  --check2    critic: best direction argmax_k Q on a 20x20 grid (self camera,
              other agent at (0,0), as plot_q_map_v6.py), circular variance,
              near-corner agreement (mixed goals), away-direction cos (avoid goals).
  --check3    for the candidate sets whose goals all pass checks 1-2:
              (i) state-free naive Bayes on 16 action-direction bins,
              (i') z3 fixed-direction cos, (ii) naive Bayes on (4x4 position
              cell, direction bin), (iii) critic IRL on the recorded vision
              vs a uniform gray image.

Usage (inside Docker, from /work/my_research/preference_inference):
    xvfb-run --auto-servernum python analyze/feasibility_goals_v7.py --rollout
    xvfb-run --auto-servernum python analyze/feasibility_goals_v7.py --check1 --check2 --check3 --git_commit <hash>
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')
sys.path.insert(0, '/work/my_research/preference_inference/analyze')
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

import irl_common_v7 as C  # noqa

SAVE_DIR = os.path.join(C.PI_ROOT, 'data', 'result', 'v7_irl', 'feasibility')
ENV_CONFIG = '/work/simulation/config/collect/self_random_other_stay.yml'
A2_ACTOR_PATH = os.path.join(C.PI_ROOT, 'data', 'model', 'v6_rl_actor_seed1_film_norelabel_curriculum.pth')
N_EP, SEQ = 200, 101
N_TRAIN = 150
LM = {'red': np.array([-9.0, 9.0]), 'green': np.array([-9.0, -9.0]),
      'blue': np.array([9.0, -9.0]), 'cyan': np.array([9.0, 9.0])}
GOALS = {
    'Red': [1, -1, 0, 0], 'Green': [-1, 1, 0, 0], 'Cyan': [0, 0, -1, 1], 'Blue': [0, 0, 1, -1],
    'Mtop': [1, 0, 0, 1], 'Mleft': [1, 1, 0, 0], 'Mbottom': [0, 1, 1, 0], 'Mright': [0, 0, 1, 1],
    'Agreen': [0, -2, 0, 0], 'Acyan': [0, 0, 0, -2],
}
SINGLE = {'Red': 'red', 'Green': 'green', 'Cyan': 'cyan', 'Blue': 'blue'}
# mixed goal -> (corner if split coord < 0, corner if >= 0, split axis 0=x 1=y)
MIXED = {'Mtop': ('red', 'cyan', 0), 'Mleft': ('green', 'red', 1),
         'Mbottom': ('green', 'blue', 0), 'Mright': ('blue', 'cyan', 1)}
AVOID = {'Agreen': 'green', 'Acyan': 'cyan'}
SETS = {'S1': ['Red', 'Cyan', 'Mtop'], 'S2': ['Mtop', 'Mleft', 'Mbottom', 'Mright'],
        'S3': ['Cyan', 'Agreen', 'Green', 'Acyan']}
# z3 directions (deg): centre -> expected end
Z3_DEG = {'Red': 135, 'Green': 225, 'Cyan': 45, 'Blue': 315, 'Mtop': 90, 'Mleft': 180,
          'Mbottom': 270, 'Mright': 0, 'Agreen': 45, 'Acyan': 225}
GRID = np.linspace(-9.0, 9.0, 20)
N_DIR_BINS = 16


def get_seed(key):
    return int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16)


def npz_path(goal):
    return os.path.join(SAVE_DIR, f'rollout_{goal}.npz')


# ---------------------------------------------------------------- rollout
def rollout():
    import creator
    from util import load_config
    from my_research.preference_inference.model.rl_agent_sac_v6 import ActorLSTM
    seed = get_seed('v7_feasibility')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = creator.create_environment(load_config(ENV_CONFIG).environment)
    env.init(); env.off_display()
    actor = ActorLSTM(film=True).to(device)
    actor.load_state_dict(torch.load(A2_ACTOR_PATH, map_location=device), strict=True)
    actor.eval()
    bound = env.world.get_boundary()
    lo, hi = np.array([bound[0][0], bound[1][0]]), np.array([bound[0][1], bound[1][1]])
    os.makedirs(SAVE_DIR, exist_ok=True)
    for goal, w in GOALS.items():
        r_t = torch.tensor(np.asarray(w, np.float32)).unsqueeze(0).to(device)
        pos = np.zeros((N_EP, SEQ, 2), np.float32)
        act = np.zeros((N_EP, SEQ, 2), np.float32)
        vis = np.zeros((N_EP, SEQ, 16, 64, 3), np.uint8)
        for n in range(N_EP):
            env.reset()
            hidden = None
            for t in range(SEQ):
                op = env.other_agent.p.copy()
                env.self_agent.step()  # A-1 random walk, as in collect_data_v7.py (its vision is not needed)
                env.world.set_camera('other')
                env.world.draw(env.self_agent, env.other_agent)
                ov = env.world.capture()
                v_t = torch.tensor(np.transpose(ov.astype(np.float32), (2, 0, 1))).unsqueeze(0).to(device)
                with torch.no_grad():
                    a_t, _, hidden = actor.sample(v_t, r_t, hidden)
                om = a_t.squeeze(0).cpu().numpy()
                env.other_agent.p = np.clip(op + om, lo, hi)
                pos[n, t], act[n, t], vis[n, t] = op, om, np.round(ov * 255).astype(np.uint8)
        np.savez_compressed(npz_path(goal), pos=pos, act=act, vis=vis, r=np.asarray(w, np.float32), seed=seed)
        print(f'rollout {goal}: saved', flush=True)


# ---------------------------------------------------------------- helpers
def ang_deg(v):
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360


def ang_diff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def circ_var(deg):
    rad = np.radians(np.asarray(deg).ravel())
    return float(1 - np.hypot(np.cos(rad).mean(), np.sin(rad).mean()))


def load(goal):
    d = np.load(npz_path(goal))
    return d['pos'], d['act'], d['vis']


# ---------------------------------------------------------------- check 1
def check1():
    out = {}
    for goal in GOALS:
        pos, _, _ = load(goal)
        p0, pf = pos[:, 0], pos[:, -1]
        dist_f = {k: np.linalg.norm(pf - v, axis=1) for k, v in LM.items()}
        e = dict(end_within5={k: float((d <= 5).mean()) for k, d in dist_f.items()})
        if goal in SINGLE:
            e['mean_final_dist'] = float(dist_f[SINGLE[goal]].mean())
            e['pass'] = e['mean_final_dist'] <= 5
        elif goal in MIXED:
            neg, posc, ax = MIXED[goal]
            near = np.where(p0[:, ax] < 0, neg, posc)
            far = np.where(p0[:, ax] < 0, posc, neg)
            m = np.abs(p0[:, ax]) >= 2
            at_near = np.array([dist_f[c][i] <= 5 for i, c in enumerate(near)])
            at_far = np.array([dist_f[c][i] <= 5 for i, c in enumerate(far)])
            e.update(n_eligible=int(m.sum()), frac_end_near=float(at_near[m].mean()),
                     frac_end_far=float(at_far[m].mean()), frac_end_neither=float((~at_near & ~at_far)[m].mean()))
            edges = [-10, -6, -2, 2, 6, 10]
            e['by_split_coord'] = {f'[{edges[i]},{edges[i+1]})': dict(
                n=int(((p0[:, ax] >= edges[i]) & (p0[:, ax] < edges[i + 1])).sum()),
                **{f'end_{c}': float((dist_f[c][(p0[:, ax] >= edges[i]) & (p0[:, ax] < edges[i + 1])] <= 5).mean())
                   for c in (neg, posc)}) for i in range(5)}
            e['pass'] = e['frac_end_near'] >= 0.8
        else:
            d = dist_f[AVOID[goal]]
            e.update(frac_end_ge20_from_avoided=float((d >= 20).mean()), mean_final_pos=pf.mean(0).tolist())
            e['pass'] = e['frac_end_ge20_from_avoided'] >= 0.8
        out[goal] = e
        print(f'[check1] {goal:8s} pass={e["pass"]}  ' + json.dumps({k: v for k, v in e.items() if k not in ('by_split_coord',)}))
    return out


# ---------------------------------------------------------------- check 2
@torch.no_grad()
def critic_best_dirs(critic, vis01, r, device, batch=1024):
    """vis01 (M,H,W,3) -> best direction index (M,) and full Q (M,64) for reward r."""
    dirs = torch.from_numpy(C.PROBE_DIRS).to(device)
    best, qs = [], []
    for i0 in range(0, len(vis01), batch):
        v = C.to_vision_tensor(vis01[i0:i0 + batch], device)
        rr = torch.tensor(np.asarray(r, np.float32)).to(device).unsqueeze(0).expand(len(v), -1)
        q = C._head(critic, C._trunk(critic, v, rr), dirs.unsqueeze(0).expand(len(v), -1, -1), rr)
        qs.append(q.cpu().numpy())
    q = np.concatenate(qs)
    return q.argmax(1), q


def grid_visions():
    import creator
    from util import load_config
    env = creator.create_environment(load_config(ENV_CONFIG).environment)
    env.init(); env.off_display()
    env.world.set_camera('self')
    env.other_agent.p = np.array([0.0, 0.0])
    xy, vis = [], []
    for y in GRID:
        for x in GRID:
            env.self_agent.p = np.array([x, y])
            env.world.draw(env.self_agent, env.other_agent)
            vis.append(env.world.capture())
            xy.append((x, y))
    return np.array(xy), np.array(vis, np.float32)


def check2(critic, device):
    xy, vis = grid_visions()
    out, maps = {}, {}
    for goal, w in GOALS.items():
        k, _ = critic_best_dirs(critic, vis, w, device)
        deg = k * 360.0 / C.K_DIRS
        maps[goal] = deg
        e = dict(circ_var=circ_var(deg))
        if goal in SINGLE:
            e['mean_cos_to_goal'] = float(np.cos(np.radians(ang_diff(deg, ang_deg(LM[SINGLE[goal]] - xy)))).mean())
            e['pass'] = True  # reference only
        elif goal in MIXED:
            neg, posc, ax = MIXED[goal]
            near = np.where(xy[:, ax] < 0, 0, 1)
            corners = np.stack([LM[neg], LM[posc]])
            d_near = ang_deg(corners[near] - xy)
            d_far = ang_deg(corners[1 - near] - xy)
            m = np.abs(xy[:, ax]) >= 3
            agree = ang_diff(deg, d_near) < ang_diff(deg, d_far)
            e.update(n_grid=int(m.sum()), near_corner_agreement=float(agree[m].mean()))
            e['pass'] = e['near_corner_agreement'] >= 0.7
        else:
            away = ang_deg(xy - LM[AVOID[goal]])
            e['mean_cos_to_away'] = float(np.cos(np.radians(ang_diff(deg, away))).mean())
            e['pass'] = e['mean_cos_to_away'] >= 0.7
        out[goal] = e
        print(f'[check2] {goal:8s} pass={e["pass"]}  ' + json.dumps(e))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 5, figsize=(20, 8.4))
    for ax, goal in zip(axs.ravel(), GOALS):
        deg = maps[goal]
        u, v = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        ax.quiver(xy[:, 0], xy[:, 1], u, v, deg, cmap='hsv', clim=(0, 360), scale=28, width=0.006)
        for name, p in LM.items():
            ax.scatter(*p, s=160, c={'red': 'r', 'green': 'g', 'blue': 'b', 'cyan': 'c'}[name], edgecolors='k')
        ax.set(xlim=(-10.5, 10.5), ylim=(-10.5, 10.5), aspect='equal',
               title=f'{goal} r={GOALS[goal]}\ncircvar={out[goal]["circ_var"]:.2f} pass={out[goal]["pass"]}')
    fig.suptitle('seed2 critic: best direction argmax_k Q(s, a_k, r) on a 20x20 grid (self camera, other at (0,0))')
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'check2_best_direction_maps.png'), dpi=100)
    return out


# ---------------------------------------------------------------- check 3
def dir_bin(act):
    return (np.floor(ang_deg(act) / (360 / N_DIR_BINS)).astype(int)) % N_DIR_BINS


def pos_cell(pos):
    c = np.clip(np.floor((pos + 10) / 5).astype(int), 0, 3)
    return c[..., 0] * 4 + c[..., 1]


def nb_scores(train_feats, test_feats, n_feat, ts):
    """train_feats: list over goals of (n, T) int features. Returns {t: (N_test_total, G) scores}."""
    logp = []
    for f in train_feats:
        cnt = np.bincount(f.ravel(), minlength=n_feat) + 1.0
        logp.append(np.log(cnt / cnt.sum()))
    logp = np.stack(logp)  # (G, n_feat)
    ll = logp[:, test_feats].transpose(1, 2, 0)  # (N, T, G)
    cum = np.cumsum(ll, axis=1)
    return {t: cum[:, t - 1] for t in ts}


def acc_at(scores, labels):
    pred, tied = C.argmax_with_ties(scores)
    return float((pred == labels).mean()), float(tied.mean())


@torch.no_grad()
def critic_q_all(critic, vis01_fn, n, unit, r_list, device, batch=2048):
    dirs = torch.from_numpy(C.PROBE_DIRS).to(device)
    q64 = np.zeros((n, len(r_list), C.K_DIRS), np.float32)
    qobs = np.zeros((n, len(r_list)), np.float32)
    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        v = C.to_vision_tensor(vis01_fn(i0, i1), device)
        acts = torch.cat([dirs.unsqueeze(0).expand(i1 - i0, -1, -1),
                          torch.from_numpy(unit[i0:i1]).to(device).unsqueeze(1)], 1)
        for j, r in enumerate(r_list):
            rr = torch.tensor(np.asarray(r, np.float32)).to(device).unsqueeze(0).expand(i1 - i0, -1)
            q = C._head(critic, C._trunk(critic, v, rr), acts, rr).cpu().numpy()
            q64[i0:i1, j], qobs[i0:i1, j] = q[:, :C.K_DIRS], q[:, C.K_DIRS]
    return q64, qobs


def check3(critic, device, eligible):
    ts = (20, 50)
    out = {}
    for sname, goals in SETS.items():
        if not all(eligible[g] for g in goals):
            out[sname] = dict(goals=goals, evaluated=False,
                              failing=[g for g in goals if not eligible[g]])
            print(f'[check3] {sname} not evaluated (failing goals: {out[sname]["failing"]})')
            continue
        data = [load(g) for g in goals]
        tr = [(p[:N_TRAIN], a[:N_TRAIN]) for p, a, _ in data]
        te_pos = np.concatenate([p[N_TRAIN:] for p, _, _ in data])
        te_act = np.concatenate([a[N_TRAIN:] for _, a, _ in data])
        te_vis = np.concatenate([v[N_TRAIN:] for _, _, v in data])
        labels = np.repeat(np.arange(len(goals)), N_EP - N_TRAIN)
        N, T = te_act.shape[:2]
        res = dict(goals=goals, evaluated=True, chance=1 / len(goals), n_test=int(N))
        s_i = nb_scores([dir_bin(a) for _, a in tr], dir_bin(te_act), N_DIR_BINS, ts)
        s_ii = nb_scores([pos_cell(p) * N_DIR_BINS + dir_bin(a) for p, a in tr],
                         pos_cell(te_pos) * N_DIR_BINS + dir_bin(te_act), 16 * N_DIR_BINS, ts)
        unit, small = C.unit_actions(te_act)
        use = ~small
        prior = np.log(np.full(len(goals), 1 / len(goals)))
        th = np.radians([Z3_DEG[g] for g in goals])
        k_ang = 2 * np.pi * np.arange(C.K_DIRS) / C.K_DIRS
        q64g = np.cos(k_ang[None, None, None, :] - th[None, None, :, None]) * np.ones((N, T, 1, 1))
        qobsg = np.cos(np.arctan2(unit[..., 1], unit[..., 0])[..., None] - th[None, None, :])
        lp_z3 = C.log_posterior_all_t(q64g, qobsg, use, C.BETA_MAIN, prior)
        r_list = [GOALS[g] for g in goals]
        flat = te_vis.reshape(N * T, *te_vis.shape[2:])
        gray = np.full((1, *te_vis.shape[2:]), 0.5, np.float32)
        lps = {}
        for cond, fn in (('true_vision', lambda i0, i1: flat[i0:i1].astype(np.float32) / 255.0),
                         ('gray', lambda i0, i1: np.repeat(gray, i1 - i0, 0))):
            q64, qobs = critic_q_all(critic, fn, N * T, unit.reshape(N * T, 2), r_list, device)
            lps[cond] = C.log_posterior_all_t(q64.reshape(N, T, len(goals), C.K_DIRS),
                                              qobs.reshape(N, T, len(goals)), use, C.BETA_MAIN, prior)
        for t in ts:
            res[f't{t}'] = dict(
                state_free_nb=acc_at(s_i[t], labels)[0], z3=acc_at(lp_z3[:, t], labels)[0],
                state_aware_nb=acc_at(s_ii[t], labels)[0],
                critic_true_vision=acc_at(lps['true_vision'][:, t], labels)[0],
                critic_gray=acc_at(lps['gray'][:, t], labels)[0])
        r50 = res['t50']
        res['gap_task'] = r50['state_aware_nb'] - r50['state_free_nb']
        res['gap_critic'] = r50['critic_true_vision'] - r50['critic_gray']
        res['state_matters'] = bool(res['gap_task'] >= 0.15 and res['gap_critic'] >= 0.15)
        out[sname] = res
        print(f'[check3] {sname} ' + json.dumps(res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollout', action='store_true')
    ap.add_argument('--check1', action='store_true')
    ap.add_argument('--check2', action='store_true')
    ap.add_argument('--check3', action='store_true')
    ap.add_argument('--git_commit', default=None)
    args = ap.parse_args()
    if args.rollout:
        rollout()
    if args.check1 or args.check2 or args.check3:
        C.set_determinism(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        critic = C.load_critic(device)
        res = dict(git_commit=args.git_commit, goals=GOALS, n_episodes=N_EP, n_train=N_TRAIN,
                   rollout_seed=get_seed('v7_feasibility'),
                   true_values_used='rollout quantities only (A-2 position/action/vision/goal); '
                                    'goal labels for evaluation and for training the naive-Bayes references')
        res['check1'] = check1()
        res['check2'] = check2(critic, device)
        eligible = {g: res['check1'][g]['pass'] and res['check2'][g]['pass'] for g in GOALS}
        res['eligible'] = eligible
        if args.check3:
            res['check3'] = check3(critic, device, eligible)
        with open(os.path.join(SAVE_DIR, 'feasibility_results.json'), 'w') as fp:
            json.dump(res, fp, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        print('Saved:', os.path.join(SAVE_DIR, 'feasibility_results.json'))


if __name__ == '__main__':
    main()
