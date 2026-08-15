import datetime
import os
import random
import subprocess
import sys

import h5py
import numpy

import torch
from config_util import load_config
from constants import (DATASET_DIR, DATASET_NAME, EXP_CONFIG_DIR, LOG_DIR_BASE,
                       MODEL_CONFIG_DIR, MODEL_DIR_BASE, RESULT_DIR,
                       SAVE_DIR_BASE, TEST_DIR_BASE)
from exp import loader, logger
import util


def gen_data_loader(data_name, batch_size, data_load_memory, device):
    data_filename = gen_data_filename(data_name)
    data = load_data(data_filename, data_load_memory)
    train_loader = loader.DataLoader(
        data['train'], batch_size, shuffle=True, device=device)

    eval_loader = loader.DataLoader(
        data['train'], batch_size, shuffle=False, device=device)

    if not ('test' in data.keys()):
        data['test'] = data['train']

    test_loader = loader.DataLoader(
        data['test'], batch_size, shuffle=False, device=device)

    return train_loader, eval_loader, test_loader


def gen_logger(log_dir):

    train_logger = logger.Logger(log_dir, 'train')
    eval_logger = logger.Logger(log_dir, 'eval')
    test_logger = logger.Logger(log_dir, 'test')
    return train_logger, eval_logger, test_logger


def load_model(model_dir, restore_epoch, model):
    model.load_state_dict(
        torch.load(model_dir + '{:05d}.pth'.format(restore_epoch))['model'])


def load_pretrain(model, exp_config):
    if not (exp_config.train.pretrain is None):
        checkpoint = torch.load(
            gen_model_dir(gen_result_dir(exp_config.train.pretrain)) +
            '{:05d}.pth'.format(exp_config.train.pretrain.epoch))['model']

        # v4 R3: SM's input width changes (m_t's 2-dim -> probe-Q's K-dim),
        # so its weights aren't shape-compatible with an exp1_l1/exp3
        # checkpoint even though the key names match. strict=False alone
        # doesn't help here -- it only tolerates missing/extra keys, not
        # shape mismatches on keys present in both, which would raise a
        # RuntimeError. Skip those, load everything else (in the normal
        # same-shape case this is identical to a plain load_state_dict).
        #
        # Special case (2026-08-14): SuperpositionModule concatenates
        # [vision(enc_out), motion] before the LSTM (see modules.py:
        # `sx = torch.cat([sv, sm], 1)`), so weight_ih's columns are
        # always [vision_cols..., motion_cols...] in that order. When the
        # column count differs (rows/gates match) ONLY the first
        # VISION_ENC_DIM columns are transferred -- NOT
        # min(old_width, new_width), which would wrongly reuse some of
        # the checkpoint's old *motion* columns (e.g. exp1_l1's 2 action
        # weight columns) as if they were meaningful for the new probe-Q
        # columns, just because they happened to land in the shared
        # numeric range. All motion/Q columns (indices >= VISION_ENC_DIM)
        # keep the model's own random init instead.
        VISION_ENC_DIM = 64
        model_state = model.state_dict()
        compatible = {}
        partial = []
        skipped = []
        for k, v in checkpoint.items():
            if k not in model_state:
                skipped.append(k)
                continue
            target = model_state[k]
            if target.shape == v.shape:
                compatible[k] = v
            elif (k.endswith('weight_ih') and target.dim() == 2 and v.dim() == 2
                  and target.shape[0] == v.shape[0]):
                n_common = min(VISION_ENC_DIM, target.shape[1], v.shape[1])
                merged = target.clone()
                merged[:, :n_common] = v[:, :n_common]
                compatible[k] = merged
                partial.append(f'{k} (kept first {n_common}/{target.shape[1]} '
                               f'vision cols only, rest random-init)')
            else:
                skipped.append(k)
        if partial:
            print(f'load_pretrain: partially loading {len(partial)} keys '
                  f'(shared prefix columns transferred, rest random-init): {partial}')
        if skipped:
            print(f'load_pretrain: skipping {len(skipped)} shape-mismatched/'
                  f'absent keys (trained from scratch instead): {skipped}')

        model.load_state_dict(compatible, strict=False)


def save_model_optimizer(model_dir, epoch, model, optimizer):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, model_dir + '{:05d}.pth'.format(epoch))


def mkdir_p(dir_path):
    subprocess.check_output(['mkdir', '-p', dir_path])


def r_print(string):
    sys.stdout.write("\r%s " % string)
    sys.stdout.flush()


def gen_data_filename(data_name):
    return DATASET_DIR + '{:s}/{:s}'.format(data_name, DATASET_NAME)


def gen_exp_config(args):
    return load_config(EXP_CONFIG_DIR + '{:s}.yml'.format(args.exp_config))


def gen_model_config(args):
    return load_config(MODEL_CONFIG_DIR + '/{:s}/{:s}.yml'.format(
        args.model.name, args.model.config))


def gen_dirs(args, test, test_name=None):
    result_dir = gen_result_dir(args)
    model_dir = gen_model_dir(result_dir)
    if test:
        assert (test_name is not None)
        test_dir = gen_test_dir(result_dir, test_name)
        log_dir = gen_log_dir(test_dir)
        save_dir = gen_save_dir(test_dir)
        return result_dir, model_dir, log_dir, save_dir
    else:
        log_dir = gen_log_dir(result_dir)
        return result_dir, model_dir, log_dir


def gen_result_dir(args):
    return RESULT_DIR + '{:s}/{:d}/'.format(args.exp_config, args.seed)


def gen_model_dir(result_dir):
    model_dir = result_dir + MODEL_DIR_BASE
    mkdir_p(model_dir)
    return model_dir


def gen_log_dir(result_dir):
    return result_dir + LOG_DIR_BASE


def gen_save_dir(result_dir):
    return result_dir + SAVE_DIR_BASE


def gen_test_dir(result_dir, test_name):
    return result_dir + TEST_DIR_BASE + test_name + '/'


def set_cudnn_config(args):
    torch.backends.cudnn.enabled = True
    if args.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def gen_optim_params(net, config):
    train_params = {}
    freezed_params = {}

    for name, param in net.named_parameters():
        param.grad = None

        if False if config.freeze is None else any(
                s in name for s in config.freeze):
            param.requres_grad = False
            freezed_params[name] = param
        else:
            param.requres_grad = True
            train_params[name] = param

    default = {'params': []}
    default.update(config.default)

    params = {}

    for pp in config.per_params:
        params[pp['name']] = {'params': []}
        params[pp['name']].update(pp['args'])

    for name, param in train_params.items():
        param.requires_grad = True

        # assert (sum([s in name for s in config.per_params.keys()]) <= 1)

        if any(pp['name'] in name for pp in config.per_params):

            for pp in config.per_params:
                if pp['name'] in name:
                    params[pp['name']]['params'].append(param)
                    # print(name, pp['name'])
                    break
        else:
            default['params'].append(param)

    print('Train params')
    for name in train_params.keys():
        print(name)
    print('')
    print('Freezed params')
    for name in freezed_params.keys():
        print(name)
    print('')

    params = list(params.values())
    params.append(default)

    return params


def seed_all(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transpose_vision(v, dim=5):
    if dim == 5:
        return v.transpose(0, 1, 4, 2, 3)
    elif dim == 4:
        return v.transpose(0, 3, 1, 2)
    elif dim == 3:
        return v.transpose(2, 0, 1)
    else:
        assert (False)


def de_transpose_vision(v, dim=5):

    if dim == 5:
        return v.transpose(0, 1, 3, 4, 2)
    elif dim == 4:
        return v.transpose(0, 2, 3, 1)
    elif dim == 3:
        return v.transpose(1, 2, 0)
    else:
        assert (False)


def scale_vision(v):
    # v4: vision may be stored as uint8 (0-255, 4x smaller on disk than
    # float32) since the source render is already 8-bit-quantized -- see
    # collect_data_r2.py. Older datasets are still float32 in [0,1];
    # both are handled here so callers don't need to know which.
    if v.dtype == numpy.uint8:
        v = v.astype(numpy.float32) / 255.0
    return v * 2 - 1


def de_scale_vision(v):
    return (v + 1) / 2


def gen_result_metadata(exp_config_name=None, seed=None, dataset_name=None):
    # v4 (2026-08-15): result JSONs previously recorded only source_h5/epoch/
    # mode, with the exp_config and weight-transfer variant only inferable
    # from the label/filename -- e.g. r3_a_direct_r2.json (original,
    # full-random-init weight_ih) vs r3_a_direct_v2_r2.json (corrected
    # partial-transfer) were indistinguishable except by reading this
    # session's chat history. Every analysis script that writes a result
    # JSON should merge this dict in, so results are self-describing even
    # after the config file or codebase moves on.
    meta = {
        'timestamp': datetime.datetime.now().isoformat(),
        'git_commit': None,
        'exp_config_name': exp_config_name,
        'config_path': None,
        'config_content': None,
        'seed': seed,
        'dataset_name': dataset_name,
    }
    # the `git` binary isn't installed in the training container (only the
    # bind-mounted .git directory is available there), so read HEAD by
    # walking .git files directly instead of shelling out to git.
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for _ in range(6):
            if os.path.isdir(os.path.join(repo_root, '.git')):
                break
            repo_root = os.path.dirname(repo_root)
        git_dir = os.path.join(repo_root, '.git')
        with open(os.path.join(git_dir, 'HEAD')) as f:
            head = f.read().strip()
        if head.startswith('ref:'):
            ref_path = os.path.join(git_dir, head[len('ref:'):].strip())
            with open(ref_path) as f:
                meta['git_commit'] = f.read().strip()
        else:
            meta['git_commit'] = head
    except Exception:
        pass
    if exp_config_name is not None:
        config_path = EXP_CONFIG_DIR + '{:s}.yml'.format(exp_config_name)
        meta['config_path'] = config_path
        try:
            with open(config_path) as f:
                meta['config_content'] = f.read()
        except Exception:
            pass
    return meta


def load_data(f_data, load_memory=False):
    f = h5py.File(f_data, 'r')

    def _recursive_load(loaded, source):

        for k, v in source.items():
            if isinstance(v, dict) or isinstance(v, h5py._hl.group.Group):
                loaded[k] = {}
                _recursive_load(loaded[k], v)
            else:
                loaded[k] = v[()] if load_memory else v

        return loaded

    data = _recursive_load({}, f)

    return data
