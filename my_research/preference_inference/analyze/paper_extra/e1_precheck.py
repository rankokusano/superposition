"""
Paper round 11 E1 (docs/paper_extra_log.md Sec.35.4; instructions Sec.3.1): before training the 50% sign-randomised
runs, check that they start from the same initial weights and draw the same training batches and motion-hiding
intervals as the same pair's 50% run (v12_s1_mixed_mse), with s3s5_precheck.py's method (round 6) unchanged:
  ep0     : the initial state_dict vs the saved 50% ep0 (v12_s1_mixed_mse/<s>/model/00000.pth), bit-wise.
  emulate : the CPU-side draws of a GPU run (batch order, hiding schedule at each init_state), no forward pass.
  run     : --nb real training steps on --device (batch indices, schedules, a1_liked of the batch, losses).
On CPU the vision masks also use the CPU generator, so emulate and run differ there (round 6 Sec.34.1); on GPU they agree.
-> data/result/paper_extra/E1_signrand50/precheck/s<s>_<device>.json (+ .npz)
Usage (inside the container): python analyze/paper_extra/e1_precheck.py --seed 0 [--device cpu] [--nb 3]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import s3s5_precheck as P  # noqa  (setup / same_sd of round 6)
import train_paper_e1  # noqa  (routes the data name paper_s1_mixed50_signrand)

OUT = os.path.join(P.RES, 'paper_extra', 'E1_signrand50', 'precheck')
EXPS = {'mixed50': 'v12_s1_mixed_mse', 'signrand50': 'paper_extra/E1_signrand50/s1_signrand50_mse'}


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
    ref0 = torch.load(os.path.join(P.RES, 'v12_s1_mixed_mse', str(a.seed), 'model', '00000.pth'), map_location='cpu')['model']
    for c, exp in EXPS.items():
        r = {}
        cfg, model, opt, runner, tl = P.setup(exp, a.seed, dev)
        sd = model.state_dict()
        r['n_keys'] = len(sd)
        r['ep0_equals_saved_50'] = P.same_sd(sd, ref0)
        it = iter(tl.load_sequence())
        for i in range(a.nb):                       # emulate: no forward pass
            _, bi = next(it)
            model.init_state(len(bi))
            arr[f'{c}_emu_idx{i}'], arr[f'{c}_emu_hid{i}'] = np.asarray(bi), model._hidden.numpy()
        cfg, model, opt, runner, tl = P.setup(exp, a.seed, dev)
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
        rec[f'{m}_same_batches'] = all(np.array_equal(arr[f'mixed50_{m}_idx{i}'], arr[f'signrand50_{m}_idx{i}']) for i in range(a.nb))
        rec[f'{m}_same_schedules'] = all(np.array_equal(arr[f'mixed50_{m}_hid{i}'], arr[f'signrand50_{m}_hid{i}']) for i in range(a.nb))
    rec['first_batches'] = [arr[f'signrand50_emu_idx{i}'].tolist() for i in range(a.nb)]
    rec['flipped_in_first_batches'] = [int((arr[f'signrand50_run_liked{i}'] == 0).sum()) for i in range(a.nb)]
    rec['hidden_steps_first_batches'] = [int(arr[f'signrand50_emu_hid{i}'].sum()) for i in range(a.nb)]
    rec['emu_equals_run_batches'] = all(np.array_equal(arr[f'signrand50_emu_idx{i}'], arr[f'signrand50_run_idx{i}']) for i in range(a.nb))
    rec['emu_equals_run_schedules'] = all(np.array_equal(arr[f'signrand50_emu_hid{i}'], arr[f'signrand50_run_hid{i}']) for i in range(a.nb))
    tag = f's{a.seed}_{a.device.replace(":", "")}'
    np.savez_compressed(os.path.join(OUT, f'{tag}.npz'), **arr)
    with open(os.path.join(OUT, f'{tag}.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k not in EXPS}),
          {c: {k: v for k, v in rec[c].items() if k != 'run_losses'} for c in EXPS}, flush=True)


if __name__ == '__main__':
    main()
