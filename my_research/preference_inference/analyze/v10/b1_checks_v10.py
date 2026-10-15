"""
v10 gate B1 checks for one trained agent (docs/v10_experiment_log.md Sec.5.3, 5.9 (2)).

1. not collapsed: v4 check_critic_health.py probes (7 actions at the Red
   reference point; action (1,0) at the 4 reference corners + centre;
   other agent at (0,0)), thresholds action_std > 0.05 and state_std > 0.05,
   in BOTH layouts (pass = both layouts pass).
2. actor reaches its colour: v6 actor_rollout_v6.py method (env.reset(),
   env.step(), actor.sample() with hidden carried over, position clipped
   to +-9.5, step as in training), 50 episodes x 100 steps, random / numpy /
   torch seeded 0 at the start of each layout. Final distance to the
   preferred colour's reference point: mean (judged, <= 5) and median.
   A-1: self camera, env self_random_other_stay, env.step() (incl. the
   RandomAgent step, as in train_rl_v3.py). A-2: other camera, env
   self_stay_other_random, A-1 stationary, no env.step() (as in
   train_rl_v3_a2.py).
3. critic best direction points at the colour: in BOTH layouts, mean angle
   difference (K=64 argmax, 21x21 grid, other agent at (0,0), excluding
   points within 2 of either reference) to the colour <= mean difference to
   the colour's place in the other layout - 30 deg.
Figures: learning curve (per-episode reward by layout, critic-1 loss),
direction/Q maps (both layouts), rollout trajectories (both layouts).
Outputs: data/result/v10_swap/b1/<label>/.
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, '/work')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, PI_DIR, git_hash  # noqa
sys.path.insert(0, PI_DIR)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

import direction_maps_v10 as dm  # noqa
from sim_v10 import creator, layout  # noqa
from sim_v10.util import load_config  # noqa
from my_research.rl_agent_sac import ActorLSTM, CriticLSTM  # noqa

DEVICE = dm.DEVICE
CFG_DIR = os.path.join(PI_DIR, 'sim_v10', 'config', 'collect')
CORNERS = {'Red_ref': (-9.0, 9.0), 'Green_ref': (-9.0, -9.0), 'Blue_ref': (9.0, -9.0),
           'Cyan_ref': (9.0, 9.0), 'Center': (0.0, 0.0)}
PROBE_ACTIONS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, -0.7)]
EPS = 0.05
N_EP, MAX_STEPS, SEED = 50, 100, 0
SPEC = {
    'a1': dict(camera='self', target='Red', env='self_random_other_stay.yml'),
    'a2': dict(camera='other', target='Green', env='self_stay_other_random.yml'),
}


def to_t(v):
    return torch.tensor(np.transpose(v.astype(np.float32), (2, 0, 1))).unsqueeze(0).to(DEVICE)


def render(env, sp, cam):
    agent, fixed = (env.self_agent, env.other_agent) if cam == 'self' else (env.other_agent, env.self_agent)
    agent.p = np.array(sp, dtype=float)
    fixed.p = np.array([0.0, 0.0])
    env.world.set_camera(cam)
    env.world.draw(env.self_agent, env.other_agent)
    return env.world.capture()


@torch.no_grad()
def health(env, critic, cam):
    out = {}
    for lo in layout.LAYOUT_NAMES:
        env.set_layout(lo)
        v = to_t(render(env, CORNERS['Red_ref'], cam))
        qa = [critic(v, torch.tensor([a], dtype=torch.float32, device=DEVICE))[0].item() for a in PROBE_ACTIONS]
        qs = {k: critic(to_t(render(env, p, cam)), torch.tensor([(1.0, 0.0)], device=DEVICE))[0].item()
              for k, p in CORNERS.items()}
        a_std, s_std = float(np.std(qa)), float(np.std(list(qs.values())))
        out[lo] = dict(action_qs=qa, corner_qs=qs, action_std=a_std, state_std=s_std,
                       healthy=bool(a_std > EPS and s_std > EPS))
    out['pass'] = all(out[lo]['healthy'] for lo in layout.LAYOUT_NAMES)
    return out


@torch.no_grad()
def rollout(env, actor, agent, lo):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env.set_layout(lo)
    tgt = layout.LANDMARKS[lo][SPEC[agent]['target']]
    old = layout.LANDMARKS[layout.other_layout(lo)][SPEC[agent]['target']]
    trajs, fin, fin_old = [], [], []
    for _ in range(N_EP):
        env.reset()
        if agent == 'a1':
            v, _, _, _, _ = env.step()
            mover = env.self_agent
        else:
            env.world.set_camera('other')
            env.world.draw(env.self_agent, env.other_agent)
            v = env.world.capture()
            mover = env.other_agent
        h, tr = None, [mover.p.copy()]
        for _ in range(MAX_STEPS):
            a, _, h = actor.sample(to_t(v), h)
            mover.p = np.clip(mover.p + a.squeeze(0).cpu().numpy(), -9.5, 9.5)
            if agent == 'a1':
                v, _, _, _, _ = env.step()
            else:
                env.world.set_camera('other')
                env.world.draw(env.self_agent, env.other_agent)
                v = env.world.capture()
            tr.append(mover.p.copy())
        tr = np.array(tr)
        trajs.append(tr)
        fin.append(np.linalg.norm(tr[-1] - tgt))
        fin_old.append(np.linalg.norm(tr[-1] - old))
    return np.array(trajs), dict(
        final_dist_colour_mean=float(np.mean(fin)), final_dist_colour_median=float(np.median(fin)),
        final_dist_colour_sd=float(np.std(fin)),
        final_dist_other_layout_place_mean=float(np.mean(fin_old)),
        final_dist_other_layout_place_median=float(np.median(fin_old)),
        pass_mean_le_5=bool(np.mean(fin) <= 5.0))


def plot_curves(log_csv, out, label):
    rows = list(csv.DictReader(open(log_csv)))
    ep = np.array([int(r['episode']) for r in rows])
    rew = np.array([float(r['total_reward']) for r in rows])
    lay = np.array([r['layout'] for r in rows])
    loss = np.array([float(r['critic1_loss_mean']) if r['critic1_loss_mean'] else np.nan for r in rows])

    def smooth(x, w=25):
        k = np.ones(w) / w
        return np.convolve(np.nan_to_num(x), k, mode='same')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for lo, c in (('original', '#4C78A8'), ('swap_rg', '#F58518')):
        m = lay == lo
        axes[0].scatter(ep[m], rew[m], s=3, c=c, alpha=0.3)
        r = np.full(len(ep), np.nan)
        r[m] = rew[m]
        idx = np.where(m)[0]
        axes[0].plot(ep[idx], np.convolve(rew[m], np.ones(25) / 25, mode='same'), c=c, lw=2, label=f'{lo} (25-ep mean)')
    axes[0].set_xlabel('episode')
    axes[0].set_ylabel('episode reward (sum over 100 steps)')
    axes[0].legend(fontsize=8)
    axes[0].set_title(f'{label}: reward by layout', fontsize=10)
    axes[1].plot(ep, loss, c='0.6', lw=0.5)
    axes[1].plot(ep, smooth(loss), c='k', lw=1.5, label='25-ep mean')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('episode')
    axes[1].set_ylabel('critic-1 MSE loss (episode mean)')
    axes[1].set_title(f'{label}: critic loss', fontsize=10)
    axes[1].legend(fontsize=8)
    fig.savefig(os.path.join(out, 'learning_curve.png'), dpi=110)
    plt.close(fig)
    half = len(ep) // 2
    return {lo: {'reward_mean_last_half': float(rew[half:][lay[half:] == lo].mean()),
                 'n_episodes': int((lay == lo).sum())} for lo in layout.LAYOUT_NAMES}


def plot_rollouts(trajs, agent, out, label):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, lo in zip(axes, layout.LAYOUT_NAMES):
        for tr in trajs[lo]:
            ax.plot(tr[:, 0], tr[:, 1], c='0.5', lw=0.6, alpha=0.7)
            ax.scatter(tr[-1, 0], tr[-1, 1], s=10, c='k', zorder=3)
        dm.draw_landmarks(ax, lo, SPEC[agent]['target'])
        ax.set_title(f'{label}, {lo}: {N_EP} rollouts x {MAX_STEPS} steps', fontsize=10)
    fig.savefig(os.path.join(out, 'rollouts.png'), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', choices=['a1', 'a2'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--critic_path', default=None, help='override (e.g. v3 reference)')
    ap.add_argument('--actor_path', default=None)
    ap.add_argument('--label', default=None)
    args = ap.parse_args()
    mdir = os.path.join(PI_DIR, 'data', 'model', 'v10')
    if args.agent == 'a1':
        cp, apth = f'v10_rl_critic_seed{args.seed}.pth', f'v10_rl_actor_seed{args.seed}.pth'
    else:
        cp, apth = f'v10_rl_a2_critic_tem2p0_eps0p0_seed{args.seed}.pth', f'v10_rl_a2_actor_tem2p0_eps0p0_seed{args.seed}.pth'
    if args.critic_path:
        cp, apth = os.path.relpath(args.critic_path, mdir), os.path.relpath(args.actor_path, mdir)
    label = args.label or f'{args.agent}_seed{args.seed}'
    out = os.path.join(OUT_ROOT, 'b1', label)
    os.makedirs(out, exist_ok=True)
    spec = SPEC[args.agent]

    env = creator.create_environment(load_config(os.path.join(CFG_DIR, spec['env'])).environment)
    env.init()
    env.off_display()
    env.reset()
    critic = CriticLSTM().to(DEVICE)
    critic.load_state_dict(torch.load(os.path.join(mdir, cp), map_location=DEVICE))
    critic.eval()
    actor = ActorLSTM().to(DEVICE)
    actor.load_state_dict(torch.load(os.path.join(mdir, apth), map_location=DEVICE))
    actor.eval()

    res = {'agent': args.agent, 'seed': args.seed, 'critic': cp, 'actor': apth, 'git_commit': git_hash()}
    res['check1_health'] = health(env, critic, spec['camera'])

    trajs = {}
    res['check2_rollout'] = {}
    for lo in layout.LAYOUT_NAMES:
        trajs[lo], res['check2_rollout'][lo] = rollout(env, actor, args.agent, lo)
    res['check2_rollout']['pass'] = all(res['check2_rollout'][lo]['pass_mean_le_5'] for lo in layout.LAYOUT_NAMES)

    maps = {}
    res['check3_direction'] = {}
    for lo in layout.LAYOUT_NAMES:
        env.set_layout(lo)
        maps[lo] = dm.compute_maps(env, critic, actor, spec['camera'])
        st = dm.angle_stats(maps[lo]['best64'], lo, spec['target'], layout.LANDMARKS)
        st_c = dm.angle_stats(maps[lo]['best64'], lo, spec['target'], layout.CUBE_CENTERS)
        st['pass_30deg'] = bool(st['mean_diff_colour'] <= st['mean_diff_place'] - 30.0)
        res['check3_direction'][lo] = {'ref_pm9': st, 'ref_cube_centre_pm13': st_c,
                                       'actor_dir_ref_pm9': dm.angle_stats(maps[lo]['actor_dir'], lo, spec['target'], layout.LANDMARKS),
                                       'qmax8_argmax_xy': [float(dm.GRID[i]) for i in np.unravel_index(np.argmax(maps[lo]['qmax8']), maps[lo]['qmax8'].shape)]}
    res['check3_direction']['pass'] = all(res['check3_direction'][lo]['ref_pm9']['pass_30deg'] for lo in layout.LAYOUT_NAMES)
    q_o, q_s = maps['original']['q8'].reshape(-1), maps['swap_rg']['q8'].reshape(-1)
    res['q8_corr_between_layouts'] = float(np.corrcoef(q_o, q_s)[0, 1])
    res['pass_all'] = res['check1_health']['pass'] and res['check2_rollout']['pass'] and res['check3_direction']['pass']

    log_csv = os.path.join(mdir, cp.replace('.pth', '_log.csv'))
    res['training'] = plot_curves(log_csv, out, label) if os.path.exists(log_csv) else None
    dm.OUT_DIR = out
    dm.plot_agent(label, dict(critic=cp, actor=apth, camera=spec['camera'], target=spec['target']), maps)
    plot_rollouts(trajs, args.agent, out, label)
    np.savez(os.path.join(out, 'b1_arrays.npz'), grid=dm.GRID,
             **{f'{lo}_{k}': v for lo in maps for k, v in maps[lo].items()},
             **{f'traj_{lo}': trajs[lo] for lo in trajs})
    json.dump(res, open(os.path.join(out, 'b1_checks.json'), 'w'), indent=2)
    print(json.dumps({k: res[k]['pass'] if isinstance(res[k], dict) and 'pass' in res[k] else res[k]
                      for k in ['check1_health', 'check2_rollout', 'check3_direction', 'pass_all']}))


if __name__ == '__main__':
    main()
