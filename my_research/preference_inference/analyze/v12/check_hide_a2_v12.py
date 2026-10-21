"""v12 check for proposal C (docs/v12_experiment_log.md Sec.43): can A-1's view be re-rendered without A-2 (A-2 moved
far outside the arena) and does it then equal the stored view except at A-2's pixels? CPU/xvfb, a few frames."""
import os, sys
import h5py, numpy as np
_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
from sim_v10 import creator, layout as L  # noqa
from sim_v10.util import load_config  # noqa
import time
cfg = load_config(os.path.join(_PI, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml'))
env = creator.create_environment(cfg.environment); env.init(); env.off_display()
g = h5py.File(os.path.join(_PI, 'data', 'data', 'v11_s2_green', 'data.h5'), 'r')['train']
sp, op, lay, sv = g['self_position'][:3], g['other_position'][:3], g['layout'][:3], g['self_vision'][:3]
def render(s, o):
    env.self_agent.p = np.asarray(s, float); env.other_agent.p = np.asarray(o, float)
    env.world.set_camera('self'); env.world.draw(env.self_agent, env.other_agent)
    return np.round(env.world.capture() * 255).astype(np.uint8)
t0 = time.time(); n = 0; same = 0; black_left = 0; diff_outside = 0
for e in range(3):
    env.set_layout(L.CODE_LAYOUT[int(lay[e])])
    for t in range(0, 101, 10):
        a = render(sp[e, t], op[e, t]); same += int(np.array_equal(a, sv[e, t]))
        b = render(sp[e, t], (1e4, 1e4))
        blk = (sv[e, t].max(-1) < 26)
        black_left += int((b.max(-1) < 26).sum())
        diff_outside += int((np.abs(b.astype(int) - sv[e, t].astype(int)).max(-1) > 0)[~blk].sum())
        n += 1
print(f'frames {n}: re-render equals stored {same}/{n}; black pixels left without A-2: {black_left}; '
      f'pixels changed outside A-2 mask: {diff_outside}; {(time.time() - t0) / (2 * n) * 1000:.1f} ms per render')
