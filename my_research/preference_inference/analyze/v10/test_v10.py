"""
v10 copy of test.py whose outputs never touch the trained model's result
directory (docs/v10_experiment_log.md Sec.5.9 (3), P2).

Differences from test.py (everything else, incl. seeding, --eval_seed and
cudnn handling, runner/saver/logger calls, is the same code path):
  - the checkpoint is read from data/result/<exp_config>/<seed>/model/
    with a plain path string (no util.gen_dirs / gen_model_dir, which run
    `mkdir -p` inside that directory);
  - torch.load uses map_location='cpu' (some protected checkpoints were
    pickled on cuda:5; v4 results_log 2026-09-07) -- values are unchanged;
  - logs and saved.h5 go to --out_dir, which must lie under
    data/result/v10_swap/.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/v10/test_v10.py --exp_config r3_a_direct_1000pretrain_400 \
        --seed 0 --device cuda:1 --test_epoch 400 \
        --test_data_name r2_a1random_a2rl_viz20 --test_batch_size 10 \
        --test_modes eval --eval_seed 0 --cudnn_deterministic \
        --save_targets state self_vision other_vision self_position other_position \
        --out_dir data/result/v10_swap/<...>
"""

import argparse
import os
import sys

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import args_util  # noqa
import model as models  # noqa
import torch  # noqa
import util  # noqa
from constants import MODEL_DIR_BASE, RESULT_DIR  # noqa
from exp import runner as runners  # noqa
from exp import saver  # noqa

V10_ROOT = os.path.realpath(os.path.join(_PI_DIR, 'data', 'result', 'v10_swap'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args_util.add_common_args(parser)
    args_util.add_test_args(parser)
    parser.add_argument('--out_dir', required=True,
                        help='output root (logs + save/saved.h5); must be under data/result/v10_swap/')
    args = parser.parse_args()

    out_dir = os.path.realpath(os.path.join(_PI_DIR, args.out_dir))
    assert out_dir.startswith(V10_ROOT + os.sep), \
        f'--out_dir must be under {V10_ROOT}, got {out_dir}'
    out_dir += '/'

    util.set_cudnn_config(args)

    util.seed_all(args.seed)

    exp_config = util.gen_exp_config(args)
    model_config = util.gen_model_config(exp_config)
    model_dir = os.path.join(_PI_DIR, RESULT_DIR, args.exp_config, str(args.seed), MODEL_DIR_BASE)
    log_dir = out_dir + 'log/'
    save_dir = out_dir + 'save/'

    if args.mask_off:
        exp_config.p_mask_vision = 0

    device = torch.device(args.device)

    model = getattr(models, exp_config.model.name)(model_config)
    model.to(device)

    ckpt = os.path.join(model_dir, '{:05d}.pth'.format(args.test_epoch))
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
    print(f'Loaded (read-only): {ckpt}')

    runner = getattr(runners, exp_config.runner)()

    _, eval_loader, test_loader = util.gen_data_loader(
        args.test_data_name, args.test_batch_size, args.data_load_memory,
        device)

    train_logger, eval_logger, test_logger = util.gen_logger(log_dir)

    eval_saver = saver.DataSaver(
        save_dir=save_dir,
        mode='eval',
        save_targets=args.save_targets,
        save_targets_index=args.save_targets_index,
        save_vision_img=args.save_vision_img,
        img_ext=args.img_ext,
    )
    test_saver = saver.DataSaver(
        save_dir=save_dir,
        mode='test',
        save_targets=args.save_targets,
        save_targets_index=args.save_targets_index,
        save_vision_img=args.save_vision_img,
        img_ext=args.img_ext,
    )

    model.eval()

    if args.eval_seed is not None:
        util.seed_all(args.eval_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    with torch.no_grad():

        if 'eval' in args.test_modes:
            runner.eval_epoch(model, args.test_epoch, exp_config,
                              eval_loader, eval_logger, eval_saver)

        if 'test' in args.test_modes:
            runner.eval_epoch(model, args.test_epoch, exp_config,
                              test_loader, test_logger, test_saver)

    print(f'Outputs: {out_dir}')
