"""
v10 stage A, items 1-3 (docs/v10_experiment_log.md Sec.5.0, 5.2, 5.9 (2)):
critic best-direction maps and Q-value maps, plus actor direction maps, for
A-1 (v3_rl_critic / v3_rl_actor, self camera, likes Red) and A-2
(v3_rl_a2_critic / v3_rl_a2_actor, other camera, likes Green), in both
layouts.

Placement as plot_q_map_v6.py: the agent whose map is drawn moves on the
grid, the other agent is fixed at (0, 0). Grid 21x21 (x, y in -10..10).
Critic: one frame, hidden=None (as probe-Q); best direction = argmax over
K=64 unit-circle actions (also K=8). Q-value map = max over the 8 probe
directions. Actor: direction of tanh(mean) (deterministic), one frame.

Angle differences (deg, circular, in [0, 180]) from each grid point to
  - the target colour's reference point in the evaluated layout ("colour")
  - the target colour's reference point in the OTHER layout ("place")
excluding grid points within 2 of either point. References (+-9, +-9);
cube centres (+-13, +-13) reported as reference only.
Pre-registered reading (critic, swap layout): colour-bound if
mean diff(colour) <= mean diff(place) - 15; place-bound if
mean diff(place) <= mean diff(colour) - 15; otherwise intermediate.

Outputs: data/result/v10_swap/stageA/direction_maps/
"""

import json
import math
import os
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

from sim_v10 import creator, layout  # noqa
from sim_v10.util import load_config  # noqa
from my_research.rl_agent_sac import ActorLSTM, CriticLSTM  # noqa

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = os.path.join(OUT_ROOT, 'stageA', 'direction_maps')
ENV_CONFIG = os.path.join(PI_DIR, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml')
MODEL_DIR = os.path.join(PI_DIR, 'data', 'model')
GRID = np.arange(-10, 11, dtype=float)
K_FINE = 64
K_PROBE = 8
EXCLUDE_R = 2.0
MARGIN_DEG = 15.0

AGENTS = {
    'A1': dict(camera='self', target='Red', critic='v3_rl_critic.pth', actor='v3_rl_actor.pth'),
    'A2': dict(camera='other', target='Green', critic='v3_rl_a2_critic.pth', actor='v3_rl_a2_actor.pth'),
}
LM_COLORS = {'Red': 'red', 'Green': 'limegreen', 'Blue': 'blue', 'Cyan': 'cyan'}


def unit_actions(k):
    a = np.array([2 * math.pi * i / k for i in range(k)])
    return a, torch.tensor(np.stack([np.cos(a), np.sin(a)], 1), dtype=torch.float32, device=DEVICE)


def to_tensor(v):
    return torch.tensor(np.transpose(v.astype(np.float32), (2, 0, 1))).unsqueeze(0).to(DEVICE)


def angdiff(a, b):
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def bearing(p, q):
    return np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0]))


def compute_maps(env, critic, actor, camera):
    ang64, act64 = unit_actions(K_FINE)
    ang8, act8 = unit_actions(K_PROBE)
    n = len(GRID)
    out = {k: np.zeros((n, n)) for k in ['best64', 'best8', 'qmax8', 'actor_dir', 'actor_mag']}
    out['q8'] = np.zeros((n, n, K_PROBE))
    for i, x in enumerate(GRID):
        for j, y in enumerate(GRID):
            mover = np.array([x, y])
            if camera == 'self':
                env.self_agent.p, env.other_agent.p = mover, np.array([0.0, 0.0])
            else:
                env.self_agent.p, env.other_agent.p = np.array([0.0, 0.0]), mover
            env.world.set_camera(camera)
            env.world.draw(env.self_agent, env.other_agent)
            v = to_tensor(env.world.capture())
            with torch.no_grad():
                q64, _ = critic(v.expand(K_FINE, -1, -1, -1), act64, hidden=None)
                q8, _ = critic(v.expand(K_PROBE, -1, -1, -1), act8, hidden=None)
                mean, _, _ = actor(v, hidden=None)
            q64 = q64.squeeze(-1).cpu().numpy()
            q8 = q8.squeeze(-1).cpu().numpy()
            a = torch.tanh(mean).squeeze(0).cpu().numpy()
            out['best64'][i, j] = np.degrees(ang64[int(np.argmax(q64))])
            out['best8'][i, j] = np.degrees(ang8[int(np.argmax(q8))])
            out['qmax8'][i, j] = q8.max()
            out['q8'][i, j] = q8
            out['actor_dir'][i, j] = np.degrees(np.arctan2(a[1], a[0]))
            out['actor_mag'][i, j] = float(np.linalg.norm(a))
    return out


def angle_stats(dir_map, lo, target, refs):
    """Mean angle difference to the target colour (this layout) and to the
    target's place in the other layout."""
    p_col = refs[lo][target]
    p_plc = refs[layout.other_layout(lo)][target]
    d_col, d_plc = [], []
    for i, x in enumerate(GRID):
        for j, y in enumerate(GRID):
            p = np.array([x, y])
            if np.linalg.norm(p - p_col) <= EXCLUDE_R or np.linalg.norm(p - p_plc) <= EXCLUDE_R:
                continue
            d_col.append(angdiff(dir_map[i, j], bearing(p, p_col)))
            d_plc.append(angdiff(dir_map[i, j], bearing(p, p_plc)))
    d_col, d_plc = np.array(d_col), np.array(d_plc)
    rad = np.radians(dir_map.reshape(-1))
    circ_var = 1.0 - float(np.hypot(np.cos(rad).mean(), np.sin(rad).mean()))
    m_c, m_p = float(d_col.mean()), float(d_plc.mean())
    if m_c <= m_p - MARGIN_DEG:
        verdict = 'colour'
    elif m_p <= m_c - MARGIN_DEG:
        verdict = 'place'
    else:
        verdict = 'intermediate'
    return {
        'n_points': int(len(d_col)),
        'mean_diff_colour': m_c, 'mean_diff_place': m_p,
        'median_diff_colour': float(np.median(d_col)), 'median_diff_place': float(np.median(d_plc)),
        'frac_within45_colour': float((d_col <= 45).mean()), 'frac_within45_place': float((d_plc <= 45).mean()),
        'circular_variance_of_direction': circ_var,
        'mode_direction_deg': float(np.bincount(np.round(((dir_map.reshape(-1) % 360) / (360 / K_FINE))).astype(int) % K_FINE).argmax() * 360 / K_FINE),
        'reading': verdict,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = load_config(ENV_CONFIG)
    env = creator.create_environment(cfg.environment)
    env.init()
    env.off_display()
    env.reset()

    results = {'git_commit': git_hash(), 'grid': '21x21, x,y in -10..10',
               'placement': 'mover on grid, other agent at (0,0) (plot_q_map_v6.py)',
               'exclusion_radius': EXCLUDE_R, 'K_fine': K_FINE, 'margin_deg': MARGIN_DEG,
               'agents': {}}
    arrays = {}
    for name, ag in AGENTS.items():
        critic = CriticLSTM().to(DEVICE)
        critic.load_state_dict(torch.load(os.path.join(MODEL_DIR, ag['critic']), map_location=DEVICE))
        critic.eval()
        actor = ActorLSTM().to(DEVICE)
        actor.load_state_dict(torch.load(os.path.join(MODEL_DIR, ag['actor']), map_location=DEVICE))
        actor.eval()
        res = {'critic': ag['critic'], 'actor': ag['actor'], 'camera': ag['camera'],
               'target': ag['target'], 'layouts': {}}
        maps = {}
        for lo in layout.LAYOUT_NAMES:
            env.set_layout(lo)
            m = compute_maps(env, critic, actor, ag['camera'])
            maps[lo] = m
            r = {}
            for key in ['best64', 'best8', 'actor_dir']:
                r[key] = {'ref_pm9': angle_stats(m[key], lo, ag['target'], layout.LANDMARKS),
                          'ref_cube_centre_pm13': angle_stats(m[key], lo, ag['target'], layout.CUBE_CENTERS)}
            r['qmax8_range'] = [float(m['qmax8'].min()), float(m['qmax8'].max())]
            r['qmax8_argmax_xy'] = [float(GRID[i]) for i in np.unravel_index(np.argmax(m['qmax8']), m['qmax8'].shape)]
            r['actor_mag_mean'] = float(m['actor_mag'].mean())
            res['layouts'][lo] = r
            for k, v in m.items():
                arrays[f'{name}_{lo}_{k}'] = v
        # same grid point, same agent: how much does the best direction move under the swap?
        res['best64_change_original_to_swap_mean_deg'] = float(
            angdiff(maps[layout.ORIGINAL]['best64'], maps[layout.SWAP_RG]['best64']).mean())
        res['actor_change_original_to_swap_mean_deg'] = float(
            angdiff(maps[layout.ORIGINAL]['actor_dir'], maps[layout.SWAP_RG]['actor_dir']).mean())
        results['agents'][name] = res
        plot_agent(name, ag, maps)

    with open(os.path.join(OUT_DIR, 'direction_maps.json'), 'w') as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(OUT_DIR, 'direction_maps.npz'), grid=GRID, **arrays)
    print(json.dumps(results, indent=2))


def draw_landmarks(ax, lo, target):
    for nm, p in layout.LANDMARKS[lo].items():
        ax.scatter(p[0], p[1], marker='D', s=70, c=LM_COLORS[nm], edgecolors='k', linewidths=0.8, zorder=5)
    p_plc = layout.LANDMARKS[layout.other_layout(lo)][target]
    ax.scatter(p_plc[0], p_plc[1], marker='o', s=180, facecolors='none', edgecolors='k',
               linestyle='--', linewidths=1.2, zorder=6)
    ax.set_xlim(-10.8, 10.8)
    ax.set_ylim(-10.8, 10.8)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_agent(name, ag, maps):
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
    X, Y = np.meshgrid(GRID, GRID, indexing='ij')
    qlo = min(maps[lo]['qmax8'].min() for lo in maps)
    qhi = max(maps[lo]['qmax8'].max() for lo in maps)
    for r, lo in enumerate(layout.LAYOUT_NAMES):
        m = maps[lo]
        ax = axes[r, 0]
        im = ax.imshow(m['qmax8'].T, origin='lower', cmap='hot', extent=[-10.5, 10.5, -10.5, 10.5],
                       vmin=qlo, vmax=qhi)
        draw_landmarks(ax, lo, ag['target'])
        ax.set_title(f'{name} critic, {lo}: Q value (max over 8 probes)', fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75)
        p_col = layout.LANDMARKS[lo][ag['target']]
        for c, key, ttl in [(1, 'best64', f'{name} critic best direction (argmax K=64)'),
                            (2, 'actor_dir', f'{name} actor direction tanh(mean)')]:
            ax = axes[r, c]
            th = np.radians(m[key])
            d_col = angdiff(m[key], np.degrees(np.arctan2(p_col[1] - Y, p_col[0] - X)))
            q = ax.quiver(X, Y, np.cos(th), np.sin(th), d_col, cmap='viridis', clim=(0, 180),
                          scale=28, width=0.004)
            draw_landmarks(ax, lo, ag['target'])
            ax.set_title(f'{ttl}, {lo}\narrow colour: angle to {ag["target"]} here; '
                         f'ring: {ag["target"]} in other layout', fontsize=9)
        fig.colorbar(q, ax=axes[r, 1:], shrink=0.75, label='angle difference to target colour (deg)')
    fig.suptitle(f'v10 stage A: {name} ({ag["critic"]}, {ag["actor"]}, {ag["camera"]} camera, likes {ag["target"]}); '
                 f'other agent at (0,0)', fontsize=12)
    fig.savefig(os.path.join(OUT_DIR, f'direction_maps_{name}.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
