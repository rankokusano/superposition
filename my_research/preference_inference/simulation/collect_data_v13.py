"""
v13 stage-1 data (docs/v13_instructions.md Sec.3.1, 4.1): the v11 stage-1
"experienced" data with A-1 following its preference on 80% of awake steps
(v11: 50%). Everything else is collect_data_v11.py itself, imported and run
unchanged (same seed key 'v11_s1', so layouts, initial A-1 positions, first
targets, A-2 positions and sleep times can be paired with v11_s1_random;
checked in analyze/v13/gate61_v13.py).

Datasets:
  v13_s1_mixed80                train 3000 / test 300 -> data/data/v13_s1_mixed80/data.h5
  v13_s1_mixed80_check_{a,b}    first 3 train episodes -> data/result/v13/gate61/

Usage (inside Docker, from the v11 worktree's my_research/preference_inference):
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_data_v13.py --dataset v13_s1_mixed80
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_data_v11 as base  # noqa

P_RULE = 0.8
GATE61 = os.path.join(base.DATA_ROOT, 'result', 'v13', 'gate61')

base.DATASET_CONFIGS['v13_s1_mixed80'] = {'a1_policy': 'mixed', 'n_data': base.FULL}
for _rep in ('a', 'b'):
    base.DATASET_CONFIGS[f'v13_s1_mixed80_check_{_rep}'] = {
        'a1_policy': 'mixed', 'n_data': {'train': 3},
        'save_path': os.path.join(GATE61, f'check_mixed80_{_rep}.h5'),
    }


if __name__ == '__main__':
    assert '--p_rule' not in sys.argv and '--seed_key' not in sys.argv, 'fixed for v13'
    assert any(a.startswith('v13_') for a in sys.argv), 'v13 datasets only'
    sys.argv += ['--p_rule', str(P_RULE)]
    base.main()
