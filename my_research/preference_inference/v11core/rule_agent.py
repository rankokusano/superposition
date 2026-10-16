"""
v11 movement policies built on top of an existing sim_v10 RandomAgent
instance (docs/v11_experiment_log.md Sec.4.5-4.9). No learning, no critic.

  random  -- the RandomAgent's own step(), unchanged (paper behaviour).
  mixed   -- each awake step: with prob p_rule (dedicated RandomState) a
             "visual rule" step, otherwise the RandomAgent's walk().

Visual rule step: unit step (speed = agent.config.v = 1) toward the
circular centroid of the liked colour's columns in the agent's own current
view. At a wall the outward component is dropped and the step renormalised
to speed 1 ("slide"); if nothing is left (corner) or the liked colour is not
visible, the RandomAgent's walk() is used instead. The disliked colour does
not enter the rule (approach only); it only enters the pleasure input.

Sleep is an agent-level state shared by both kinds of step: while asleep the
agent does not move (whatever the coin says, no coin is drawn); the sleep
onset draw (python `random`, as in AgentBase.step) happens after every awake
step. The RandomAgent goal check (re-draw target when within goal_margin)
runs after every awake step so walk() never overshoots its target.

Per-step source codes (`a1_source`):
    0 random step, 1 rule step, 2 rule turn but fell back to a random step,
    3 asleep.
liked / disliked colours and the camera are arguments so stage 2 can reuse
this for A-2 (liked = Green).
"""

import random

import numpy as np

from .pleasure import color_direction

SRC_RANDOM, SRC_RULE, SRC_RULE_FALLBACK, SRC_SLEEP = 0, 1, 2, 3


def slide_step(p, d, bounds, speed, eps=1e-9):
    """Return a feasible step of length `speed` along d (outward components
    removed), or None if no component is left."""
    d = np.asarray(d, dtype=np.float64) * speed
    for _ in range(3):
        out = False
        for k in range(2):
            lo, hi = bounds[k]
            if p[k] + d[k] > hi + eps or p[k] + d[k] < lo - eps:
                d[k] = 0.0
                out = True
        n = np.linalg.norm(d)
        if n < 1e-9:
            return None
        d = d / n * speed
        if not out:
            return d
    return None


class AgentPolicy(object):
    def __init__(self, agent, policy, p_rule=0.5, liked='Red', rng=None):
        assert policy in ('random', 'mixed'), policy
        self.agent = agent
        self.policy = policy
        self.p_rule = p_rule
        self.liked = liked
        self.rng = rng

    def set_rng(self, rng):
        self.rng = rng

    def step(self, vision):
        """vision: the agent's own current view (16, 64, 3) in [0,1] (taken
        at the current position, before moving). Returns (motion, source)."""
        a = self.agent
        if self.policy == 'random':
            src = SRC_SLEEP if a.is_sleep else SRC_RANDOM
            return a.step(), src

        if a.is_sleep:
            a.sleep_count -= 1
            if a.sleep_count == 0:
                a.is_sleep = False
            return np.array([0., 0.]), SRC_SLEEP

        use_rule = self.rng.rand() < self.p_rule
        m = None
        if use_rule:
            d = color_direction(vision, self.liked)
            if d is not None:
                d = slide_step(a.p, d, a.world.get_boundary(), a.config.v)
            if d is not None:
                a.p = a.p + d
                m, src = d, SRC_RULE
        if m is None:
            m = np.array(a.walk())
            src = SRC_RULE_FALLBACK if use_rule else SRC_RANDOM
        if a.is_goal():
            a.reset_target()
        if random.random() < a.config.sleep_p:
            a.is_sleep = True
            a.sleep_count = a.config.sleep_base + \
                random.randrange(-a.config.sleep_margin, a.config.sleep_margin + 1)
        return m, src
