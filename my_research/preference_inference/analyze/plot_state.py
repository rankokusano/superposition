"""
Plot PCA of SM hidden states, colored by:
  - self_position / other_position (2D colormap, as original)
  - a2_landmark (4-class colors: Red/Green/Blue/Yellow)

Usage (from analyze/ dir, inside Docker /work):
    cd /work/my_research/preference_inference/analyze
    python plot_state.py \
        --result_dir ../data/result/new_exp_a/seed0 \
        --epoch 200 \
        --mode eval \
        --plot a2_landmark \
        --plot_layer other
"""

import argparse
import os
import subprocess

import numpy

from analyzer import PCAAnalyzer
from data_loader import DataLoader
from visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', required=True,
                        help='path to experiment result dir containing saved.h5')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--dim_x', type=int, default=0)
    parser.add_argument('--dim_y', type=int, default=1)
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--state_type', default='hidden')
    parser.add_argument('--plot', type=str, default='a2_landmark',
                        choices=['self_position', 'other_position', 'a2_landmark'])
    parser.add_argument('--plot_layer', type=str, default='other',
                        help='which SM hidden state to visualize (self/other)')
    parser.add_argument('--analyze_layer', type=str, default='other')
    parser.add_argument('--analyze_mode', type=str, default='eval')
    parser.add_argument('--pca_components', type=int, default=2)
    parser.add_argument('--field_size', type=int, default=10)
    args = parser.parse_args()

    saved_h5 = os.path.join(args.result_dir, 'saved.h5')
    loader = DataLoader()
    loader.set_data(saved_h5, args.epoch)

    visualizer = Visualizer(args.field_size)
    label_prefix = 'PC'

    savedir_root = os.path.join(
        args.result_dir,
        '{epoch:05d}/{mode}/state/map_by_{analyze_layer}_{analyze_mode}/'.format(**vars(args)))
    savedir_plot = os.path.join(savedir_root, '{plot_layer}/{plot}/'.format(**vars(args)))
    pcadir = os.path.join(
        args.result_dir,
        '{epoch:05d}/pca/{analyze_layer}_{analyze_mode}/'.format(**vars(args)))

    os.makedirs(savedir_plot, exist_ok=True)
    os.makedirs(pcadir, exist_ok=True)

    pca_name = os.path.join(pcadir, 'pca.pkl')
    analyzer = PCAAnalyzer(n_components=args.pca_components)

    is_base = not os.path.exists(pca_name)
    if is_base:
        analyzer.fit(
            loader.get_flatten_hidden(
                mode=args.analyze_mode,
                layer=args.analyze_layer,
                state_type=args.state_type))
        analyzer.save(pca_name)
        numpy.savetxt(
            os.path.join(pcadir, 'pca_contribution.txt'),
            analyzer.contribution_ratio,
            fmt='%.3f')
    else:
        analyzer.load(pca_name)

    h = analyzer.transform_sequence(
        loader.get_hidden(args.mode, args.plot_layer, args.state_type))

    dim_x = args.dim_x
    dim_y = args.dim_y

    # Position coloring
    if args.plot in ['self_position', 'other_position']:
        position = loader.get_value([args.mode, args.plot, 'truth'])
        landmark_labels = None
    else:
        # a2_landmark: position not used for color; still load for shape
        position = loader.get_value([args.mode, 'other_position', 'truth'])
        landmark_labels = loader.get_a2_landmark_label(args.mode)

    visualizer.init_plot()
    visualizer.put_label(
        label_prefix + str(dim_x + 1),
        label_prefix + str(dim_y + 1),
    )

    visualizer.plot(
        h[:, :, [dim_x, dim_y]],
        position,
        args.plot,
        analyzer.get_lim([dim_x, dim_y]),
        landmark_labels=landmark_labels,
    )

    save_name = os.path.join(savedir_plot, '{:d}_{:d}'.format(dim_x, dim_y))
    visualizer.save_png(save_name)
    visualizer.save_eps(save_name)
    print(f"Saved: {save_name}.png")
