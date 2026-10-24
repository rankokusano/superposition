"""
v15 movement policy (docs/v15_experiment_log.md Sec.2): the v11 "mixed"
policy (v11core/rule_agent.py) with the liked colour replaced by the
current target = the most liked colour that is still available and has a
positive value; the target is eaten when the agent comes within r_eat of
the arena corner nearest its cube. With no target left the agent walks at
random (the coin is still drawn every awake step, so random-number use does
not depend on the preference until a choice differs).

Per-step source codes: v11's 0 random, 1 rule, 2 rule -> random fallback,
3 asleep, plus 4 = no target (random step).
step() returns (motion, source, target, eaten): target is the colour the
step was taken for (None if none), eaten the colour eaten after this step
(None if none); the caller applies world.eat() after all agents moved.
"""

import random

import numpy as np

from v11core.pleasure import color_direction
from v11core.rule_agent import (SRC_RANDOM, SRC_RULE, SRC_RULE_FALLBACK, SRC_SLEEP,
                                AgentPolicy, slide_step)
from .world import CORNER_POINT

SRC_NO_TARGET = 4
R_EAT = 1.5


def current_target(values, rank, available):
    for c in rank:
        if available[c] and values[c] > 0:
            return c
    return None


class PrefPolicy(AgentPolicy):
    def __init__(self, agent, policy, p_rule=0.8, rank=None, values=None, rng=None, r_eat=R_EAT):
        super().__init__(agent, policy, p_rule=p_rule, liked=None, rng=rng)
        self.rank, self.values, self.r_eat = rank, values, r_eat

    def set_preference(self, rank, values):
        self.rank, self.values = rank, values

    def step(self, vision, world):
        """vision: the agent's own view at the current position."""
        a = self.agent
        if self.policy == 'random':
            src = SRC_SLEEP if a.is_sleep else SRC_RANDOM
            return a.step(), src, None, None

        target = current_target(self.values, self.rank, world.available)
        if a.is_sleep:
            a.sleep_count -= 1
            if a.sleep_count == 0:
                a.is_sleep = False
            return np.array([0., 0.]), SRC_SLEEP, target, None

        use_rule = self.rng.rand() < self.p_rule
        m = None
        if use_rule and target is not None:
            d = color_direction(vision, target)
            if d is not None:
                d = slide_step(a.p, d, a.world.get_boundary(), a.config.v)
            if d is not None:
                a.p = a.p + d
                m, src = d, SRC_RULE
        if m is None:
            m = np.array(a.walk())
            src = SRC_RANDOM if not use_rule else (SRC_NO_TARGET if target is None else SRC_RULE_FALLBACK)
        if a.is_goal():
            a.reset_target()
        eaten = None
        if target is not None and np.linalg.norm(a.p - CORNER_POINT[world.corner_of[target]]) <= self.r_eat:
            eaten = target
        if random.random() < a.config.sleep_p:
            a.is_sleep = True
            a.sleep_count = a.config.sleep_base + \
                random.randrange(-a.config.sleep_margin, a.config.sleep_margin + 1)
        return m, src, target, eaten
