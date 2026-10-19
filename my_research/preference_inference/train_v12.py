"""
v12 copy of train_v11.py (docs/v12_experiment_log.md Sec.4): identical except
that the model class is looked up in model.model_v12, then model.model_v11 (so v12 classes
need no edit of the existing model/__init__.py). Run from the v11 worktree's
my_research/preference_inference (all data/result paths are relative).
"""
import argparse
import os
import sys

# Ensure this package's modules take priority over root-level packages
_PI_DIR = os.path.dirname(os.path.abspath(__file__))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import torch
import torch.optim as optim

import args_util
import model as models
from model import model_v11, model_v12
from exp import runner as runners
import util

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args_util.add_common_args(parser)
    # v11 6.2 continuation (docs/v11_experiment_log.md Sec.7.6): resume model
    # AND optimizer state from <resume_exp>/<seed>/model/<resume_epoch>.pth and
    # continue at resume_epoch + 1 (load_pretrain restores weights only, which
    # would reset AdamW's moments). Outputs go to this run's own exp dir.
    parser.add_argument('--resume_exp', default=None)
    parser.add_argument('--resume_epoch', type=int, default=None)
    args = parser.parse_args()

    util.set_cudnn_config(args)

    util.seed_all(args.seed)

    exp_config = util.gen_exp_config(args)
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(args, test=False)

    device = torch.device(args.device)

    model_cls = getattr(model_v12, exp_config.model.name, None) or \
        getattr(model_v11, exp_config.model.name, None) or \
        getattr(models, exp_config.model.name)
    model = model_cls(model_config)
    model.to(device)

    util.load_pretrain(model, exp_config)

    params = util.gen_optim_params(model, exp_config.train.optim_config)

    optimizer = getattr(optim, exp_config.train.optimizer)(params)

    runner = getattr(runners, exp_config.runner)()

    train_loader, eval_loader, test_loader = util.gen_data_loader(
        exp_config.data.name, exp_config.train.batch_size,
        args.data_load_memory, device)

    train_logger, eval_logger, test_logger = util.gen_logger(log_dir)

    start_epoch = 1
    if args.resume_exp is not None:
        assert exp_config.train.pretrain is None
        ck_path = os.path.join('data', 'result', args.resume_exp, str(args.seed), 'model',
                               '{:05d}.pth'.format(args.resume_epoch))
        ck = torch.load(ck_path, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = args.resume_epoch + 1
        print(f'Resumed model+optimizer from {args.resume_exp} seed {args.seed} epoch {args.resume_epoch}')
    else:
        util.save_model_optimizer(model_dir, 0, model, optimizer)
    for epoch in range(start_epoch, exp_config.train.max_epochs + 1):

        runner.train_epoch(
            model,
            optimizer,
            epoch,
            exp_config,
            train_loader,
            train_logger,
        )

        if epoch % exp_config.train.save_interval == 0:
            util.save_model_optimizer(model_dir, epoch, model, optimizer)

        if epoch % exp_config.train.test_interval == 0:
            with torch.no_grad():

                runner.eval_epoch(
                    model,
                    epoch,
                    exp_config,
                    eval_loader,
                    eval_logger,
                )

                runner.eval_epoch(
                    model,
                    epoch,
                    exp_config,
                    test_loader,
                    test_logger,
                )
