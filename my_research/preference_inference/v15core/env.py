"""
v15 environment factory: sim_v10's Environment wiring (agents from the
sim_v10 configs, the paper's world config) with World15 as the world.
"""

import os

from sim_v10 import creator
from sim_v10.util import load_config
from .world import World15

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD_CONFIG = os.path.join(_PI_DIR, 'sim_v10', 'config', 'world', 'default.yml')


class Env15(object):
    def __init__(self, self_agent='random', other_agent='random'):
        self.world = World15(config=load_config(WORLD_CONFIG))
        self.self_agent = creator.create_agent(self_agent)
        self.other_agent = creator.create_agent(other_agent)
        self.self_agent.assign_world(self.world)
        self.other_agent.assign_world(self.world)

    def init(self):
        self.world.init()
        self.world.off_display()

    def reset(self):
        self.self_agent.reset()
        self.other_agent.reset()

    def views(self):
        """Both agents' views at the current positions (uint8-quantized float)."""
        w = self.world
        w.set_camera('self'); w.draw(self.self_agent, self.other_agent)
        sv = w.capture()
        w.set_camera('other'); w.draw(self.self_agent, self.other_agent)
        ov = w.capture()
        return sv, ov
