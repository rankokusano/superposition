"""
v15 (D) (docs/v15_experiment_log.md Sec.33): eval_s2_hide_v15.py run unchanged, with the NS classes (model/model_v15ns.py)
registered in model_v15's namespace (eval_s2_v15.load looks classes up there). Same arguments as eval_s2_hide_v15.py.
"""
import os
import runpy
import sys

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

from model import model_v15, model_v15ns  # noqa: E402
for _name in ('SuperpositionNetworkMotionPleasureHoldV15NS', 'SuperpositionNetworkMotionPleasureHoldOtherV15NS'):
    setattr(model_v15, _name, getattr(model_v15ns, _name))

if __name__ == '__main__':
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_s2_hide_v15.py'), run_name='__main__')
