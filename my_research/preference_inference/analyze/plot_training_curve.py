"""
Training curve plotter (2026-08-22), per v4_experiment_log.md Sec.8.4/10.

Added after discovering that R3-A(1000pretrain)'s loss curves show a
"long plateau -> sharp late breakthrough" pattern (feature_prediction_other
slowly WORSENED for 160 epochs then dropped 2.3x in the last ~30) that is
completely invisible if you only look at the final-epoch number -- R3-stay
appeared to have "worse prediction than R3-A despite an easier (static)
target" until the full curve revealed R3-stay just hadn't reached its own
breakthrough yet at epoch 200. From now on, every training run should have
its curve actually plotted before its final-epoch numbers are trusted as
converged (Sec.10 checklist).

Reads log/{train,eval,test}/{self_vision,feature_prediction_self,
feature_prediction_other}.log (plain "epoch, value" CSV, written by
exp/logger.py) for a given exp_config and plots them together.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/plot_training_curve.py --exp_config r3_stay_400
    python analyze/plot_training_curve.py --exp_config r3_stay_400 r3_a_direct_1000pretrain exp1_l1_1000 --label r3_stay_400_vs_r3a_vs_paper
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = ['self_vision', 'feature_prediction_self', 'feature_prediction_other']
SPLITS = ['train', 'eval']

# root-repo exp configs live outside my_research/preference_inference/data/result
ROOT_RESULT_DIR = '/work/data/result'
LOCAL_RESULT_DIR = 'data/result'
SAVE_DIR = 'data/result/baseline_v4'


def find_log_dir(exp_config, seed=0):
    local = os.path.join(LOCAL_RESULT_DIR, exp_config, str(seed), 'log')
    if os.path.isdir(local):
        return local
    root = os.path.join(ROOT_RESULT_DIR, exp_config, str(seed), 'log')
    if os.path.isdir(root):
        return root
    raise FileNotFoundError(f'no log dir found for {exp_config} (checked {local}, {root})')


def load_curve(log_dir, split, metric):
    path = os.path.join(log_dir, split, f'{metric}.log')
    if not os.path.exists(path):
        return [], []
    epochs, values = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e, v = line.split(',')
            epochs.append(int(e))
            values.append(float(v))
    return epochs, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', nargs='+', required=True,
                         help='one or more exp_config names to overlay')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--label', default=None)
    args = parser.parse_args()
    label = args.label or '_vs_'.join(args.exp_config)

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 4 * len(METRICS)), sharex=False)
    if len(METRICS) == 1:
        axes = [axes]

    for exp_config in args.exp_config:
        log_dir = find_log_dir(exp_config, args.seed)
        for ax, metric in zip(axes, METRICS):
            for split, style in zip(SPLITS, ['-', '--']):
                epochs, values = load_curve(log_dir, split, metric)
                if not epochs:
                    continue
                ax.plot(epochs, values, style, label=f'{exp_config} ({split})', alpha=0.8)

    for ax, metric in zip(axes, METRICS):
        ax.set_title(metric)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{label}_training_curve.png')
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f'Saved: {out_path}')

    # also print a quick numeric summary: has each curve's eval self_vision
    # loss changed by >5% in its last 20% of epochs vs the preceding 20%
    # (a cheap automatic "still moving late?" flag)
    print('\n=== late-training movement check (eval/self_vision) ===')
    for exp_config in args.exp_config:
        log_dir = find_log_dir(exp_config, args.seed)
        epochs, values = load_curve(log_dir, 'eval', 'self_vision')
        if len(epochs) < 5:
            print(f'{exp_config}: not enough eval points to check')
            continue
        n = len(values)
        early = values[max(0, n // 2 - n // 5):n // 2]
        late = values[-max(1, n // 5):]
        if not early or not late:
            continue
        early_mean = sum(early) / len(early)
        late_mean = sum(late) / len(late)
        pct = 100 * (early_mean - late_mean) / early_mean if early_mean else 0
        flag = '*** still moving late -- check convergence ***' if abs(pct) > 5 else 'looks converged'
        print(f'{exp_config}: mid-training mean={early_mean:.3f}  final-20% mean={late_mean:.3f}  '
              f'change={pct:+.1f}%  {flag}')


if __name__ == '__main__':
    main()
