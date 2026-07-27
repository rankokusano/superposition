"""
Data collection for preference inference experiment.

A-1: SAC RL agent (ActorLSTM + CriticLSTM loaded from pre-trained models)
A-2: One of the following behaviors per dataset:
    - new_self_rl_other_cycler:  A-2 cycles Red → Green → Blue → Yellow (→ Red ...)
    - new_self_rl_other_stay:    A-2 stays still (StayAgent-like, random initial pos)
    - new_self_rl_other_random:  A-2 moves to random targets
    - new_self_rl_other_red:     A-2 always goes to Red landmark (preference=Red)
    (and similarly for Green, Blue, Yellow single-preference variants)

Saved to h5 datasets:
    {mode}/self_vision, self_motion, self_position,
    other_motion, other_position,
    a1_q_values, a2_target_landmark

For grid mode (Q-value heatmap), saves:
    {mode}/self_vision, self_position, other_position, a1_q_values

Usage (inside Docker /work):
    python my_research/preference_inference/simulation/collect_data.py \
        --dataset new_self_rl_other_cycler

The script detects environment by checking /work existence.
"""

import argparse
import os
import sys
import random

import h5py
import numpy as np
import torch

# ── path setup: allow importing from /work/simulation and /work/my_research ──
WORK_ROOT = '/work' if os.path.isdir('/work') else os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.insert(0, WORK_ROOT)                                # for my_research.rl_agent_sac
sys.path.insert(0, os.path.join(WORK_ROOT, 'simulation'))  # for creator, environment (must come first)

import creator  # noqa: E402 (from /work/simulation/creator.py)
from my_research.rl_agent_sac import ActorLSTM, CriticLSTM  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────

VISION_CHANNELS = 3
MOTION_DIM = 2

LANDMARK_POSITIONS = {
    'Red':    np.array([-9.0,  9.0]),
    'Green':  np.array([-9.0, -9.0]),
    'Blue':   np.array([ 9.0, -9.0]),
    'Yellow': np.array([ 9.0,  9.0]),
}
LANDMARK_NAMES = ['Red', 'Green', 'Blue', 'Yellow']
LANDMARK_ID = {name: i for i, name in enumerate(LANDMARK_NAMES)}

ACTOR_PATH  = os.path.join(WORK_ROOT, 'my_research', 'rl_model_v6_actor.pth')
CRITIC_PATH = os.path.join(WORK_ROOT, 'my_research', 'rl_model_v6_critic.pth')

V2_ACTOR_PATH  = os.path.join(WORK_ROOT, 'my_research', 'preference_inference',
                               'data', 'model', 'v2_rl_actor.pth')
V2_CRITIC_PATH = os.path.join(WORK_ROOT, 'my_research', 'preference_inference',
                               'data', 'model', 'v2_rl_critic.pth')

# ── dataset configurations ───────────────────────────────────────────────────

DATASET_CONFIGS = {
    'new_self_rl_other_cycler': {
        'a2_mode': 'cycler',
        'seq_length': 200,
        'n_data': {'train': 500, 'test': 50},
        'grid': False,
    },
    'new_self_rl_other_stay': {
        'a2_mode': 'stay',
        'seq_length': 200,
        'n_data': {'train': 500, 'test': 50},
        'grid': False,
    },
    'new_self_rl_other_random': {
        'a2_mode': 'random',
        'seq_length': 200,
        'n_data': {'train': 500, 'test': 50},
        'grid': False,
    },
    # Single-preference A-2 variants
    'new_self_rl_other_red':    {'a2_mode': 'fixed', 'target': 'Red',    'seq_length': 200, 'n_data': {'train': 500, 'test': 50}, 'grid': False},
    'new_self_rl_other_green':  {'a2_mode': 'fixed', 'target': 'Green',  'seq_length': 200, 'n_data': {'train': 500, 'test': 50}, 'grid': False},
    'new_self_rl_other_blue':   {'a2_mode': 'fixed', 'target': 'Blue',   'seq_length': 200, 'n_data': {'train': 500, 'test': 50}, 'grid': False},
    'new_self_rl_other_yellow': {'a2_mode': 'fixed', 'target': 'Yellow', 'seq_length': 200, 'n_data': {'train': 500, 'test': 50}, 'grid': False},
    # Grid: for Q-value heatmap analysis
    'new_grid_q_map': {
        'a2_mode': 'fixed', 'target': 'Red',
        'grid': True,
        'num_div': 20,
    },
    # ── v2 datasets (use v2_rl_actor.pth trained with red +1, green -2) ──────
    'v2_self_rl_other_stay': {
        'a2_mode': 'stay',
        'seq_length': 200,
        'n_data': {'train': 100, 'test': 50},
        'grid': False,
        'use_v2_rl': True,
    },
    'v2_self_rl_other_random_landmark': {
        'a2_mode': 'random_landmark',
        'seq_length': 200,
        'n_data': {'train': 500, 'test': 50},
        'grid': False,
        'use_v2_rl': True,
    },
    'v2_fixed_red': {
        'a2_mode': 'fixed', 'target': 'Red',
        'seq_length': 200,
        'n_data': {'train': 100, 'test': 50},
        'grid': False,
        'use_v2_rl': True,
    },
    'v2_fixed_green': {
        'a2_mode': 'fixed', 'target': 'Green',
        'seq_length': 200,
        'n_data': {'train': 100, 'test': 50},
        'grid': False,
        'use_v2_rl': True,
    },
    'v2_fixed_blue': {
        'a2_mode': 'fixed', 'target': 'Blue',
        'seq_length': 200,
        'n_data': {'train': 100, 'test': 50},
        'grid': False,
        'use_v2_rl': True,
    },
    'v2_fixed_yellow': {
        'a2_mode': 'fixed', 'target': 'Yellow',
        'seq_length': 200,
        'n_data': {'train': 100, 'test': 50},
        'grid': False,
        'use_v2_rl': True,
    },
    # v2 grid: A-1 Q-value landscape using v2 RL agent
    'v2_grid_q_map': {
        'a2_mode': 'fixed', 'target': 'Red',
        'grid': True,
        'num_div': 20,
        'use_v2_rl': True,
    },
    # ── Additional validation test datasets (use existing trained models) ──
    # RandomLandmark: A-2 visits landmarks in random order → breaks cycler pattern
    'test_self_rl_other_random': {
        'a2_mode': 'random_landmark',
        'seq_length': 200,
        'n_data': {'train': 100, 'test': 50},
        'grid': False,
    },
    # Fixed-preference A-2: always heading to one landmark
    'test_self_rl_other_fixed_red': {
        'a2_mode': 'fixed', 'target': 'Red',
        'seq_length': 200,
        'n_data': {'train': 100, 'test': 50},
        'grid': False,
    },
    'test_self_rl_other_fixed_green': {
        'a2_mode': 'fixed', 'target': 'Green',
        'seq_length': 200,
        'n_data': {'train': 100, 'test': 50},
        'grid': False,
    },
    'test_self_rl_other_fixed_blue': {
        'a2_mode': 'fixed', 'target': 'Blue',
        'seq_length': 200,
        'n_data': {'train': 100, 'test': 50},
        'grid': False,
    },
    'test_self_rl_other_fixed_yellow': {
        'a2_mode': 'fixed', 'target': 'Yellow',
        'seq_length': 200,
        'n_data': {'train': 100, 'test': 50},
        'grid': False,
    },
}

# ── A-2 behaviour classes ────────────────────────────────────────────────────

GOAL_MARGIN = 1.5
SPEED = 1.0


class CyclerA2:
    """A-2 cycles through all 4 landmarks in fixed order indefinitely."""

    def __init__(self):
        self._order = LANDMARK_NAMES[:]
        self._idx = 0
        self._target_name = None
        self._target_pos = None

    def reset(self):
        self._idx = 0
        self._target_name = self._order[0]
        self._target_pos = LANDMARK_POSITIONS[self._target_name].copy()

    def get_action_and_label(self, current_pos):
        direction = self._target_pos - current_pos
        distance = np.linalg.norm(direction)
        if distance < GOAL_MARGIN:
            # Advance to next landmark
            self._idx = (self._idx + 1) % len(self._order)
            self._target_name = self._order[self._idx]
            self._target_pos = LANDMARK_POSITIONS[self._target_name].copy()
            direction = self._target_pos - current_pos
            distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return np.array([0.0, 0.0]), LANDMARK_ID[self._target_name]
        action = direction / distance * SPEED
        return action, LANDMARK_ID[self._target_name]


class StayA2:
    """A-2 stays at its initial position. 'Target' = nearest landmark at step 0."""

    def reset(self, pos):
        dists = {name: np.linalg.norm(LANDMARK_POSITIONS[name] - pos)
                 for name in LANDMARK_NAMES}
        self._nearest = min(dists, key=dists.get)

    def get_action_and_label(self, current_pos):
        return np.array([0.0, 0.0]), LANDMARK_ID[self._nearest]


class RandomA2:
    """A-2 moves to random positions. Label = nearest landmark at each step."""

    def __init__(self, world):
        self._world = world
        self._target = None

    def reset(self):
        self._pick_new_target()

    def _pick_new_target(self):
        bound = self._world.get_boundary()
        x = random.uniform(bound[0][0], bound[0][1])
        y = random.uniform(bound[1][0], bound[1][1])
        self._target = np.array([x, y])

    def get_action_and_label(self, current_pos):
        direction = self._target - current_pos
        distance = np.linalg.norm(direction)
        if distance < GOAL_MARGIN:
            self._pick_new_target()
            direction = self._target - current_pos
            distance = np.linalg.norm(direction)
        if distance < 1e-6:
            action = np.array([0.0, 0.0])
        else:
            action = direction / distance * SPEED
        dists = {name: np.linalg.norm(LANDMARK_POSITIONS[name] - current_pos)
                 for name in LANDMARK_NAMES}
        nearest = min(dists, key=dists.get)
        return action, LANDMARK_ID[nearest]


class FixedA2:
    """A-2 always heads for a fixed landmark."""

    def __init__(self, target_name):
        self._name = target_name
        self._pos = LANDMARK_POSITIONS[target_name].copy()
        self._label = LANDMARK_ID[target_name]

    def reset(self):
        pass

    def get_action_and_label(self, current_pos):
        direction = self._pos - current_pos
        distance = np.linalg.norm(direction)
        if distance < GOAL_MARGIN:
            return np.array([0.0, 0.0]), self._label
        return direction / distance * SPEED, self._label


class RandomLandmarkA2:
    """A-2 moves to randomly chosen landmarks — no fixed order.

    On each arrival, the next landmark is chosen uniformly at random from
    the remaining three (so it always changes target), breaking the fixed
    Red→Green→Blue→Yellow sequence of CyclerA2.
    """

    def __init__(self):
        self._target_name = None
        self._target_pos = None

    def reset(self):
        self._target_name = random.choice(LANDMARK_NAMES)
        self._target_pos = LANDMARK_POSITIONS[self._target_name].copy()

    def get_action_and_label(self, current_pos):
        direction = self._target_pos - current_pos
        distance = np.linalg.norm(direction)
        if distance < GOAL_MARGIN:
            others = [n for n in LANDMARK_NAMES if n != self._target_name]
            self._target_name = random.choice(others)
            self._target_pos = LANDMARK_POSITIONS[self._target_name].copy()
            direction = self._target_pos - current_pos
            distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return np.array([0.0, 0.0]), LANDMARK_ID[self._target_name]
        return direction / distance * SPEED, LANDMARK_ID[self._target_name]


# ── RL agent helpers ─────────────────────────────────────────────────────────

def load_rl_agents(device):
    actor = ActorLSTM().to(device)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    actor.eval()

    critic = CriticLSTM().to(device)
    critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))
    critic.eval()

    return actor, critic


def vision_to_tensor(vision_np, device):
    """Convert HxWxC uint8/float HWC → 1x3xHxW float tensor."""
    v = vision_np.astype(np.float32) / 255.0
    v = np.transpose(v, (2, 0, 1))  # HWC → CHW
    return torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)


def get_rl_action_and_q(actor, critic, vision_np, device):
    """
    Stateless call (hidden=None) per step for simplicity.
    Returns action (2-dim np.array), q_value (float scalar).
    """
    v_t = vision_to_tensor(vision_np, device)
    with torch.no_grad():
        action_t, _, _ = actor.sample(v_t, hidden=None)
        q_t, _ = critic(v_t, action_t, hidden=None)
    action = action_t.squeeze(0).cpu().numpy()
    q_value = q_t.squeeze().cpu().item()
    return action, q_value


# ── v2 RL agent helpers ──────────────────────────────────────────────────────

def load_v2_rl_agents(device):
    """Load v2 RL agents trained with red +1, green -2 reward."""
    actor = ActorLSTM().to(device)
    actor.load_state_dict(torch.load(V2_ACTOR_PATH, map_location=device))
    actor.eval()
    critic = CriticLSTM().to(device)
    critic.load_state_dict(torch.load(V2_CRITIC_PATH, map_location=device))
    critic.eval()
    return actor, critic


def v2_vision_to_tensor(vision_np, device):
    """Float32 [0,1] HWC → 1x3xHxW tensor without /255 (GL_FLOAT output)."""
    v = vision_np.astype(np.float32)
    v = np.transpose(v, (2, 0, 1))
    return torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)


def get_v2_rl_action_and_q(actor, critic, vision_np, device):
    """Stateless action+Q using v2 RL agent (trained with [0,1] float input)."""
    v_t = v2_vision_to_tensor(vision_np, device)
    with torch.no_grad():
        action_t, _, _ = actor.sample(v_t, hidden=None)
        q_t, _ = critic(v_t, action_t, hidden=None)
    action = action_t.squeeze(0).cpu().numpy()
    q_value = q_t.squeeze().cpu().item()
    return action, q_value


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='new_self_rl_other_cycler')
    parser.add_argument('--display', action='store_true', default=False)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = creator.create_environment()

    if not args.display:
        env.off_display()
    env.init()
    env.set_camera('self')

    # Load RL agents (v2 datasets use the new v2 RL model)
    if cfg.get('use_v2_rl', False):
        actor, critic = load_v2_rl_agents(device)
        action_fn = get_v2_rl_action_and_q
    else:
        actor, critic = load_rl_agents(device)
        action_fn = get_rl_action_and_q

    # Prepare save path
    save_dir = os.path.join(WORK_ROOT, 'my_research', 'preference_inference',
                            'data', 'data', args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'data.h5')
    h5_file = h5py.File(save_path, 'w')

    if cfg['grid']:
        _collect_grid(env, actor, critic, cfg, h5_file, device, action_fn=action_fn)
    else:
        _collect_sequential(env, actor, critic, cfg, h5_file, args.dataset, device,
                            action_fn=action_fn)

    h5_file.close()
    print(f"Saved to {save_path}")


def _collect_grid(env, actor, critic, cfg, h5_file, device, action_fn=None):
    """Grid collection: for each (A-1 pos, A-2 pos) combination, record Q-value."""
    if action_fn is None:
        action_fn = get_rl_action_and_q
    world = env.world
    xmin, xmax = world.config.xmin, world.config.xmax
    ymin, ymax = world.config.ymin, world.config.ymax
    nd = cfg['num_div']
    N = nd * nd

    cam_h = world.config.camera.agent.height
    cam_w = world.config.camera.agent.width

    v_shape = (N, N, cam_h, cam_w, VISION_CHANNELS)
    p_shape = (N, N, MOTION_DIM)
    q_shape = (N, N, 1)

    mode = 'train'
    data = {
        'self_vision':   h5_file.create_dataset(mode + '/self_vision',   v_shape, dtype='f'),
        'self_position': h5_file.create_dataset(mode + '/self_position', p_shape, dtype='f'),
        'other_position':h5_file.create_dataset(mode + '/other_position',p_shape, dtype='f'),
        'a1_q_values':   h5_file.create_dataset(mode + '/a1_q_values',   q_shape, dtype='f'),
    }

    import numpy as np
    xs = np.linspace(xmin, xmax, nd)
    ys = np.linspace(ymin, ymax, nd)

    n = 0
    for ox in xs:
        for oy in ys:
            op = np.array([ox, oy])
            t = 0
            for sx in xs:
                for sy in ys:
                    sp = np.array([sx, sy])
                    env.set_agent_pos(sp, op)
                    env.set_camera('self')
                    sv = env.capture()

                    action, q = action_fn(actor, critic, sv, device)

                    data['self_vision'][n, t] = sv
                    data['self_position'][n, t] = sp
                    data['other_position'][n, t] = op
                    data['a1_q_values'][n, t] = np.array([q], dtype=np.float32)
                    t += 1
            n += 1
            if n % 50 == 0:
                print(f"  grid {n}/{N} done")


def _make_a2(mode_cfg, world):
    a2_mode = mode_cfg['a2_mode']
    if a2_mode == 'cycler':
        return CyclerA2()
    elif a2_mode == 'stay':
        return StayA2()
    elif a2_mode == 'random':
        return RandomA2(world)
    elif a2_mode == 'fixed':
        return FixedA2(mode_cfg['target'])
    elif a2_mode == 'random_landmark':
        return RandomLandmarkA2()
    else:
        raise ValueError(f"Unknown a2_mode: {a2_mode}")


def _collect_sequential(env, actor, critic, cfg, h5_file, dataset_name, device, action_fn=None):
    if action_fn is None:
        action_fn = get_rl_action_and_q
    world = env.world
    seq_length = cfg['seq_length']
    cam_h = world.config.camera.agent.height
    cam_w = world.config.camera.agent.width

    a2_agent = _make_a2(cfg, world)

    for mode, n_data in cfg['n_data'].items():
        print(f"Collecting {mode}: {n_data} episodes x {seq_length} steps")
        m_shape = (n_data, seq_length, MOTION_DIM)
        p_shape = m_shape
        v_shape = (n_data, seq_length, cam_h, cam_w, VISION_CHANNELS)
        q_shape = (n_data, seq_length, 1)
        l_shape = (n_data, seq_length)

        data = {
            'self_vision':        h5_file.create_dataset(mode + '/self_vision',        v_shape, dtype='f'),
            'self_motion':        h5_file.create_dataset(mode + '/self_motion',        m_shape, dtype='f'),
            'self_position':      h5_file.create_dataset(mode + '/self_position',      p_shape, dtype='f'),
            'other_motion':       h5_file.create_dataset(mode + '/other_motion',       m_shape, dtype='f'),
            'other_position':     h5_file.create_dataset(mode + '/other_position',     p_shape, dtype='f'),
            'a1_q_values':        h5_file.create_dataset(mode + '/a1_q_values',        q_shape, dtype='f'),
            'a2_target_landmark': h5_file.create_dataset(mode + '/a2_target_landmark', l_shape, dtype='i'),
        }

        for n in range(n_data):
            env.reset()
            op = env.other_agent.get_position().copy()

            # Reset A-2 behaviour
            if isinstance(a2_agent, StayA2):
                a2_agent.reset(op)
            elif isinstance(a2_agent, CyclerA2):
                a2_agent.reset()
            elif isinstance(a2_agent, RandomA2):
                a2_agent.reset()
            else:
                a2_agent.reset()

            for t in range(seq_length):
                # Capture A-1's visual observation
                env.set_camera('self')
                sv = env.capture()

                sp = env.self_agent.get_position().copy()
                op = env.other_agent.get_position().copy()

                # A-1: RL agent (stateless)
                a1_action, q_val = action_fn(actor, critic, sv, device)

                # A-2: behaviour agent
                a2_action, a2_label = a2_agent.get_action_and_label(op)

                # Apply actions to environment
                new_sp = sp + a1_action
                new_op = op + a2_action

                # Clip to world bounds
                bound = world.get_boundary()
                new_sp = np.clip(new_sp, [bound[0][0], bound[1][0]], [bound[0][1], bound[1][1]])
                new_op = np.clip(new_op, [bound[0][0], bound[1][0]], [bound[0][1], bound[1][1]])

                env.set_agent_pos(new_sp, new_op)

                # Save current step data
                data['self_vision'][n, t]        = sv
                data['self_motion'][n, t]         = a1_action
                data['self_position'][n, t]       = sp
                data['other_motion'][n, t]        = a2_action
                data['other_position'][n, t]      = op
                data['a1_q_values'][n, t]         = np.array([q_val], dtype=np.float32)
                data['a2_target_landmark'][n, t]  = a2_label

            print(f"  {mode} ep {n+1}/{n_data} done")


if __name__ == '__main__':
    main()
