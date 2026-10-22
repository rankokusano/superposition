"""
v14-C stage-1 data (docs/v14_instructions.md Sec.1.3, 3.3): the v11 stage-1 "experienced" data (50% rule
steps) with A-1's liked colour drawn per episode, Red or Green with prob. 1/2, from its own sequence
RandomState(md5('v14c/a1_liked')) over the full train+test episode order (as the layout sequence, and
independent of it). Everything else is collect_data_v11.py itself, imported and run unchanged (same seed key
'v11_s1'), so layouts, initial A-1 positions, first targets, A-2 positions and sleep times pair with
v11_s1_random / v11_s1_mixed; Red-liked episodes should be identical to v11_s1_mixed. The rule approaches the
episode's liked colour; the pleasure sign (liked - disliked) is applied in the model from the saved
per-episode `a1_liked` (0 Green, 1 Red, the a2_liked code) and `a1_liked_t` (per step, for the loader).

Datasets:
  v14_s1_mixedpref               train 3000 / test 300 -> data/data/v14_s1_mixedpref/data.h5
  v14_s1_mixedpref_check_{a,b}   first 3 train episodes -> data/result/v14/datacheck/

Usage (inside Docker, from the v11 worktree's my_research/preference_inference):
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_data_v14.py --dataset v14_s1_mixedpref
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_data_v11 as base  # noqa
from v11core.rule_agent import AgentPolicy  # noqa

LIKED_KEY = 'v14c/a1_liked'
CHECK = os.path.join(base.DATA_ROOT, 'result', 'v14', 'datacheck')

base.DATASET_CONFIGS['v14_s1_mixedpref'] = {'a1_policy': 'mixed', 'n_data': base.FULL}
for _rep in ('a', 'b'):
    base.DATASET_CONFIGS[f'v14_s1_mixedpref_check_{_rep}'] = {
        'a1_policy': 'mixed', 'n_data': {'train': 3},
        'save_path': os.path.join(CHECK, f'check_mixedpref_{_rep}.h5'),
    }


def episode_liked(n_total):
    """1 = Red, 0 = Green per episode over the full train+test order (own RandomState)."""
    rng = np.random.RandomState(base.get_seed(LIKED_KEY))
    return (rng.rand(n_total) < 0.5).astype(np.uint8)


class PerEpisodeLikedPolicy(AgentPolicy):
    """set_rng() is called once at the start of every episode (collect_data_v11.collect); the liked colour
    of that episode is set there. The coin, python `random` and numpy draws are untouched."""

    def __init__(self, *a, liked_seq=None, **k):
        super().__init__(*a, **k)
        self.liked_seq = liked_seq
        self.ep = 0

    def set_rng(self, rng):
        super().set_rng(rng)
        self.liked = 'Red' if self.liked_seq[self.ep] == 1 else 'Green'
        self.ep += 1


def main():
    assert any(a.startswith('v14_') for a in sys.argv), 'v14 datasets only'
    assert '--p_rule' not in sys.argv and '--seed_key' not in sys.argv and '--liked' not in sys.argv
    liked = episode_liked(sum(base.FULL.values()))
    base.AgentPolicy = lambda agent, policy, p_rule, liked_=None, **k: PerEpisodeLikedPolicy(
        agent, policy, p_rule=p_rule, liked_seq=liked)
    base.main()
    ds = sys.argv[sys.argv.index('--dataset') + 1]
    cfg = base.DATASET_CONFIGS[ds]
    path = os.path.realpath(cfg.get('save_path') or os.path.join(base.DATA_ROOT, 'data', ds, 'data.h5'))
    with h5py.File(path, 'a') as f:
        f.attrs['a1_liked'] = 'per episode: dataset a1_liked (0 Green, 1 Red), RandomState(md5(%r))' % LIKED_KEY
        i0 = 0
        for split, n in base.FULL.items():   # same global episode order as the layouts
            if split in cfg['n_data']:
                m = cfg['n_data'][split]
                f.create_dataset(f'{split}/a1_liked', data=liked[i0:i0 + m])
                f.create_dataset(f'{split}/a1_liked_t', data=np.repeat(liked[i0:i0 + m, None], 101, 1))
            i0 += n
    print('a1_liked written', path)


if __name__ == '__main__':
    main()
