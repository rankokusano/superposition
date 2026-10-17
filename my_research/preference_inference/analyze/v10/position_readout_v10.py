"""
v10 (Sec.5.32): layout read-out from the TRUE positions of A-2 and of A-1 on a
dataset's test split, with the B4 method (standardised, C=1.0, 5-fold by
episode, folds RandomState(0), t = 1,5,10,20,50,100). Checks that A-2's
position does not reveal the layout before the main gates are computed.
"""
import argparse, json, os, sys
import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, PI_DIR, git_hash  # noqa
from layout_readout_v10 import TIMES, folds  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    f = h5py.File(os.path.join(PI_DIR, 'data', 'data', args.dataset, 'data.h5'), 'r')['test']
    lay = f['layout'][()].astype(int)
    op, sp = f['other_position'][()], f['self_position'][()]

    def acc(X):
        pred = np.zeros(len(lay))
        for tr, te in folds(np.arange(len(lay))):
            pred[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=10000)).fit(X[tr], lay[tr]).predict(X[te])
        return float((pred == lay).mean())
    res = {'dataset': args.dataset, 'git_commit': git_hash(), 'n_original': int((lay == 0).sum()), 'n_swap': int((lay == 1).sum()),
           'A2_position': {}, 'A1_position': {}, 'A2_mean_xy_by_layout': {}}
    for t in TIMES:
        res['A2_position'][str(t)] = acc(op[:, t - 1])
        res['A1_position'][str(t)] = acc(sp[:, t - 1])
        res['A2_mean_xy_by_layout'][str(t)] = [op[lay == 0, t - 1].mean(0).tolist(), op[lay == 1, t - 1].mean(0).tolist()]
    res['A2_terminal_mean_xy_by_layout'] = [op[lay == 0, -1].mean(0).tolist(), op[lay == 1, -1].mean(0).tolist()]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=2)
    print(json.dumps({k: res[k] for k in ('dataset', 'n_original', 'n_swap', 'A2_position', 'A1_position', 'A2_terminal_mean_xy_by_layout')}))


if __name__ == '__main__':
    main()
