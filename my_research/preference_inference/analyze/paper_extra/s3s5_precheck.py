"""
Paper round 6 (docs/paper_extra_log.md Sec.34; paper/check/paper_instructions_round6.md Sec.1.3): before training the
sign-randomised runs of pairs 4-6 (s3-s5), check that they start from the same initial weights and draw the same
training batches and motion-hiding intervals as the same pair's 80% run (v13_s1_mixed_mse).
train_v12.py's setup is rebuilt (seed_all -> model -> encoder load -> optimizer -> loaders; nothing is written under
the runs' result directories) for both experiment configs:
  ep0     : the initial state_dict, compared bit-wise with the saved 80% ep0 (v13_s1_mixed_mse/<s>/model/00000.pth)
            and with the dry run (paper_extra/C_signrand/dry_mse/<s>/model/00000.pth, train_paper_c.py, max_epochs 0).
  emulate : the CPU-side draws of a GPU run only (numpy permutation -> batch order; torch CPU generator -> the
            schedule at each init_state, after the model init). On GPU, vision masks and dropout use the CUDA
            generator (model/util.mask, F.dropout), so these are the draws of the GPU run. No forward pass.
  run     : --nb real training steps (runner.run_seq, backward, AdamW step) on --device, recording the batch
            indices, the schedules, the batch's a1_liked_t and the losses.
-> data/result/paper_extra/C_signrand/precheck_s3s5/s<s>_<device>.json (+ .npz)
Usage (inside the container): python analyze/paper_extra/s3s5_precheck.py --seed 3 [--device cpu] [--nb 3]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI)
sys.path.insert(0, _HERE)
import train_paper_c  # noqa  (routes util.gen_data_loader: signrand data name -> v13_s1_mixed80 + signs.h5)
import util  # noqa
from model import model_v11, model_v12, model_v14  # noqa
import model as models  # noqa
from exp import runner as runners  # noqa

RES = os.path.join(_PI, 'data', 'result')
OUT = os.path.join(RES, 'paper_extra', 'C_signrand', 'precheck_s3s5')
EXPS = {'mixed80': 'v13_s1_mixed_mse', 'signrand': 'paper_extra/C_signrand/s1_signrand_mse'}


def setup(exp, seed, device):
    """train_v12.py up to the first train_epoch, without gen_dirs / loggers / saving."""
    util.seed_all(seed)
    cfg = util.load_config(util.EXP_CONFIG_DIR + f'{exp}.yml')
    mcfg = util.gen_model_config(cfg)
    cls = getattr(model_v12, cfg.model.name, None) or getattr(model_v11, cfg.model.name, None) or \
        getattr(model_v14, cfg.model.name, None) or getattr(models, cfg.model.name)
    model = cls(mcfg)
    model.to(device)
    ck = torch.load(cfg.train.pretrain_encoder_from, map_location='cpu')['model']
    keep = {k: v for k, v in ck.items()
            if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module', 'share_lns'))}
    assert not model.load_state_dict(keep, strict=False).unexpected_keys
    opt = getattr(optim, cfg.train.optimizer)(util.gen_optim_params(model, cfg.train.optim_config))
    runner = getattr(runners, cfg.runner)()
    tl, _, _ = util.gen_data_loader(cfg.data.name, cfg.train.batch_size, False, device)
    return cfg, model, opt, runner, tl


def same_sd(a, b):
    return sorted(a) == sorted(b) and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--nb', type=int, default=3)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(OUT, exist_ok=True)
    rec, arr = {}, {}
    ref0 = torch.load(os.path.join(RES, 'v13_s1_mixed_mse', str(a.seed), 'model', '00000.pth'), map_location='cpu')['model']
    dry = os.path.join(RES, 'paper_extra', 'C_signrand', 'dry_mse', str(a.seed), 'model', '00000.pth')
    dry0 = torch.load(dry, map_location='cpu')['model'] if os.path.exists(dry) else None
    for c, exp in EXPS.items():
        r = {}
        cfg, model, opt, runner, tl = setup(exp, a.seed, dev)
        sd = model.state_dict()
        r['n_keys'] = len(sd)
        r['ep0_equals_saved_80'] = same_sd(sd, ref0)
        r['ep0_equals_dry'] = None if dry0 is None else same_sd(sd, dry0)
        it = iter(tl.load_sequence())
        for i in range(a.nb):                       # emulate: no forward pass
            _, bi = next(it)
            model.init_state(len(bi))
            arr[f'{c}_emu_idx{i}'], arr[f'{c}_emu_hid{i}'] = np.asarray(bi), model._hidden.numpy()
        cfg, model, opt, runner, tl = setup(exp, a.seed, dev)
        losses = []
        for i, (seq, bi) in enumerate(runner.data_iterator(1, tl)):
            liked = seq.data.get('a1_liked_t')
            loss, _ = runner.run_seq(model, seq, 1, cfg)
            opt.zero_grad()
            sum(loss.values()).backward()
            opt.step()
            arr[f'{c}_run_idx{i}'], arr[f'{c}_run_hid{i}'] = np.asarray(bi), model._hidden.numpy()
            if liked is not None:
                arr[f'{c}_run_liked{i}'] = liked[:, 0].cpu().numpy()
            losses.append({k: float(v) for k, v in loss.items()})
            if i + 1 == a.nb:
                break
        r['run_losses'] = losses
        rec[c] = r
    for m in ('emu', 'run'):
        rec[f'{m}_same_batches'] = all(np.array_equal(arr[f'mixed80_{m}_idx{i}'], arr[f'signrand_{m}_idx{i}']) for i in range(a.nb))
        rec[f'{m}_same_schedules'] = all(np.array_equal(arr[f'mixed80_{m}_hid{i}'], arr[f'signrand_{m}_hid{i}']) for i in range(a.nb))
    rec['first_batches'] = [arr[f'signrand_emu_idx{i}'].tolist() for i in range(a.nb)]
    rec['flipped_in_first_batches'] = [int((arr[f'signrand_run_liked{i}'] == 0).sum()) for i in range(a.nb)]
    rec['hidden_steps_first_batches'] = [int(arr[f'signrand_emu_hid{i}'].sum()) for i in range(a.nb)]
    rec['emu_equals_run_batches'] = all(np.array_equal(arr[f'signrand_emu_idx{i}'], arr[f'signrand_run_idx{i}']) for i in range(a.nb))
    rec['emu_equals_run_schedules'] = all(np.array_equal(arr[f'signrand_emu_hid{i}'], arr[f'signrand_run_hid{i}']) for i in range(a.nb))
    tag = f's{a.seed}_{a.device.replace(":", "")}'
    np.savez_compressed(os.path.join(OUT, f'{tag}.npz'), **arr)
    with open(os.path.join(OUT, f'{tag}.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k not in ('mixed80', 'signrand')}),
          {c: {k: v for k, v in rec[c].items() if k != 'run_losses'} for c in EXPS})


if __name__ == '__main__':
    main()
