"""
v4 R2 MG (Motion Generator) prediction accuracy.

Unlike v3 (where MG's output only fed VE, never touched SM, so its R^2
being catastrophic (-576000) was diagnostically meaningless), R2 uses
exp3's original architecture where MG's output IS SM's process-2 input.
This is the first point where MG's accuracy actually matters for anything
downstream, and where the Sec.5.2 speed-scale fix (0.15x -> 1.0x) should
show up as resolving the v3-era scale mismatch.

Runs a lightweight forward pass (no vision saved) over the eval set,
collecting pred['other_motion'] (MG's output) vs the true other_motion.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/eval_mg_r2.py
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import r2_score

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import model as models  # noqa
import util  # noqa

EXP_CONFIG_NAME = 'r2_a2_rl'
SEED = 0
EPOCH = 200
TEST_DATA_NAME = 'r2_a1random_a2rl'
BATCH_SIZE = 10
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SAVE_DIR = 'data/result/baseline_v4'


class Args:
    exp_config = EXP_CONFIG_NAME
    seed = SEED


def main():
    exp_config = util.gen_exp_config(Args())
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(Args(), test=False)

    model = getattr(models, exp_config.model.name)(model_config)
    model.to(DEVICE)
    util.load_model(model_dir, EPOCH, model)
    model.eval()

    _, eval_loader, _ = util.gen_data_loader(
        TEST_DATA_NAME, BATCH_SIZE, False, DEVICE)

    all_pred = []
    all_truth = []

    with torch.no_grad():
        for seq_loader, batch_index in eval_loader.load_sequence():
            model.init_state(BATCH_SIZE)
            for x, truth, t in seq_loader:
                # matches MotionGenerationPredictionRunner.gen_mask_prob:
                # both self/other vision visible only at t==0, masked after
                p_mask = 0 if t == 0 else 1
                pred = model(x, p_mask_vision_self=p_mask, p_mask_vision_other=p_mask)
                all_pred.append(pred['other_motion'].cpu().numpy())
                all_truth.append(truth['other_motion'].cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    truth = np.concatenate(all_truth, axis=0)

    print(f'n_samples={len(pred)}')
    print(f'true motion range: x=[{truth[:,0].min():.3f}, {truth[:,0].max():.3f}]  '
          f'y=[{truth[:,1].min():.3f}, {truth[:,1].max():.3f}]')
    print(f'pred motion range: x=[{pred[:,0].min():.3f}, {pred[:,0].max():.3f}]  '
          f'y=[{pred[:,1].min():.3f}, {pred[:,1].max():.3f}]')

    r2_x = r2_score(truth[:, 0], pred[:, 0])
    r2_y = r2_score(truth[:, 1], pred[:, 1])
    r2_joint = r2_score(truth, pred)
    mse_x = float(np.mean((truth[:, 0] - pred[:, 0]) ** 2))
    mse_y = float(np.mean((truth[:, 1] - pred[:, 1]) ** 2))

    print(f'R^2 (x-axis): {r2_x:.4f}')
    print(f'R^2 (y-axis): {r2_y:.4f}')
    print(f'R^2 (joint):  {r2_joint:.4f}')
    print(f'MSE (x): {mse_x:.6f}')
    print(f'MSE (y): {mse_y:.6f}')

    result = {
        'n_samples': int(len(pred)),
        'true_range_x': [float(truth[:, 0].min()), float(truth[:, 0].max())],
        'true_range_y': [float(truth[:, 1].min()), float(truth[:, 1].max())],
        'pred_range_x': [float(pred[:, 0].min()), float(pred[:, 0].max())],
        'pred_range_y': [float(pred[:, 1].min()), float(pred[:, 1].max())],
        'r2_x': float(r2_x), 'r2_y': float(r2_y), 'r2_joint': float(r2_joint),
        'mse_x': mse_x, 'mse_y': mse_y,
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, 'r2_mg_accuracy.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
