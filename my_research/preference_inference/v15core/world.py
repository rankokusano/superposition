"""
v15 world (docs/v15_experiment_log.md Sec.2, 4.1): the sim_v10 World with
the field split into floor/ground + 4 corner cubes (v15core/make_models.py),
so that per episode the red/green layout and one initially gray colour are
set, and a cube turns gray when an agent eats it. Nothing else changes
(cube geometry, materials, lighting, cameras, arena). Draw order is
self, field base, cubes (TL, BL, BR, TR), other.

State: `available` = {colour: bool}; a gray cube stays gray for the rest of
the episode. The state changes only via set_episode() (between episodes)
and eat() (when an agent eats).
"""

import os

import numpy as np

from sim_v10 import layout
from sim_v10.simulation.obj import Obj
from sim_v10.simulation.objloader import ObjLoader
from sim_v10.simulation.world import World

COLORS = ('Red', 'Green', 'Blue', 'Cyan')
COLOR_CODE = {c: i for i, c in enumerate(COLORS)}
CORNERS = ('TL', 'BL', 'BR', 'TR')
# arena corner nearest to each cube (cubes occupy [10, 16]^2 outside the arena)
CORNER_POINT = {'TL': np.array([-10., 10.]), 'BL': np.array([-10., -10.]),
                'BR': np.array([10., -10.]), 'TR': np.array([10., 10.])}
CUBE_CENTER = {'TL': np.array([-13., 13.]), 'BL': np.array([-13., -13.]),
               'BR': np.array([13., -13.]), 'TR': np.array([13., 13.])}
CORNER_COLORS = {'TL': ('Red', 'Green'), 'BL': ('Red', 'Green'), 'BR': ('Blue',), 'TR': ('Cyan',)}
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource', 'model')


def color_corners(layout_name):
    """{colour: corner} for a sim_v10 layout name."""
    tl_red = layout.LANDMARKS[layout_name]['Red'][1] > 0
    return {'Red': 'TL' if tl_red else 'BL', 'Green': 'BL' if tl_red else 'TL',
            'Blue': 'BR', 'Cyan': 'TR'}


class World15(World):
    def init(self):
        self.gl_init(self.config.camera.agent.width, self.config.camera.agent.height)
        sim_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'sim_v10', 'simulation', 'resource', 'model')
        self.obj_loader = ObjLoader()
        self.objects = []

        def obj(model, visible=True, y=0):
            return Obj(model=model, translation=(0, y, 0), rotation=(0, 0, 1, 0), visible=visible)

        self.self = obj(self.obj_loader.load_obj(os.path.join(sim_dir, self.config.model.self)),
                        y=self.config.eye_level)
        self.objects.append(self.self)
        self.field = obj(self.obj_loader.load_obj(os.path.join(MODEL_DIR, 'field_base.obj')))
        self.objects.append(self.field)
        self.cubes = {}
        for corner in CORNERS:
            for color in CORNER_COLORS[corner] + ('Gray',):
                o = obj(self.obj_loader.load_obj(os.path.join(MODEL_DIR, f'cube_{corner}_{color}.obj')),
                        visible=False)
                self.cubes[corner, color] = o
                self.objects.append(o)
        self.other_model = self.obj_loader.load_obj(os.path.join(sim_dir, self.config.model.other))
        self.other = obj(self.other_model, y=self.config.eye_level)
        self.objects.append(self.other)

        self.boundary = ((self.config.xmin, self.config.xmax), (self.config.ymin, self.config.ymax))
        self._World__init_camera()
        self.set_episode(layout.ORIGINAL, None)

    def set_episode(self, layout_name, gray=None):
        """Between episodes: red/green layout and the initially gray colour
        (None = all present; v15 data always give one colour)."""
        self.layout = layout_name
        self.corner_of = color_corners(layout_name)
        self.available = {c: c != gray for c in COLORS}
        self._update_cubes()

    def set_layout(self, name):
        self.set_episode(name, None)

    def eat(self, color):
        assert self.available[color], color
        self.available[color] = False
        self._update_cubes()

    def _update_cubes(self):
        for o in self.cubes.values():
            o.set_visible(False)
        for color, corner in self.corner_of.items():
            self.cubes[corner, color if self.available[color] else 'Gray'].set_visible(True)

    def available_vec(self):
        return np.array([self.available[c] for c in COLORS], np.uint8)
