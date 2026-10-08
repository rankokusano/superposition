"""
fig5_v6_base pre-flight: does the Autoencoder actually receive v6's trained
Encoder, or does load_state_dict_strict=False let it silently stay at random
init? Mirrors train.py exactly (gen_exp_config -> build model -> load_pretrain)
and compares every self_vision_encoder_module / other_vision_encoder_module /
share_lns parameter against the v6_s3_base_l1 ep200 and v6_s2_base_mse ep400
checkpoints with torch.equal. Also checks each parameter DIFFERED from its
random init before loading, so a match cannot be vacuous.

With --ckpt_dir/--epochs it instead checks checkpoints that train.py actually
saved (e.g. fig5_v6_base 00000/00010/00100), since the encoder is frozen there.

Exit code 1 if any parameter fails to match.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/verify_fig5_v6_encoder_load.py
    python analyze/verify_fig5_v6_encoder_load.py --saved_exp fig5_v6_base --epochs 0 10 100
"""
import argparse
import os
import sys

import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import model as models  # noqa
import util  # noqa

PREFIXES = ('self_vision_encoder_module.', 'other_vision_encoder_module.', 'share_lns.')
REFS = [('v6_s3_base_l1', 200), ('v6_s2_base_mse', 400)]


class Args:
    def __init__(self, exp_config, seed=0):
        self.exp_config = exp_config
        self.seed = seed


def ckpt_state(exp_config, epoch, seed=0):
    _, model_dir, _ = util.gen_dirs(Args(exp_config, seed), test=False)
    return torch.load(model_dir + '{:05d}.pth'.format(epoch), map_location='cpu')['model']


def enc_keys(state):
    return sorted(k for k in state if k.startswith(PREFIXES))


def compare(name, state, refs):
    ok = True
    keys = enc_keys(state)
    print(f'--- {name}: {len(keys)} encoder/share_lns tensors ---')
    for rname, rstate in refs.items():
        rkeys = enc_keys(rstate)
        missing = [k for k in keys if k not in rstate]
        n_eq = sum(1 for k in keys if k in rstate and torch.equal(state[k].cpu(), rstate[k].cpu()))
        status = 'ALL EQUAL' if (n_eq == len(keys) and not missing and len(rkeys) == len(keys)) else 'MISMATCH'
        if status != 'ALL EQUAL':
            ok = False
        print(f'  vs {rname}: {n_eq}/{len(keys)} torch.equal, ref has {len(rkeys)} keys, '
              f'missing-in-ref={len(missing)} -> {status}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', default='fig5_v6_base')
    ap.add_argument('--saved_exp', default=None,
                    help='check checkpoints saved by train.py for this exp instead of a fresh load')
    ap.add_argument('--epochs', type=int, nargs='*', default=[])
    args = ap.parse_args()

    refs = {f'{e} ep{ep}': ckpt_state(e, ep) for e, ep in REFS}
    all_ok = True

    # S2 vs S3: the encoder was frozen during S3, so these must match each other.
    r = list(refs.values())
    s3_s2 = all(torch.equal(r[0][k], r[1][k]) for k in enc_keys(r[0]))
    print(f'v6_s3_base_l1 ep200 vs v6_s2_base_mse ep400 encoder/share_lns: '
          f'{"ALL EQUAL" if s3_s2 else "DIFFERENT"} ({len(enc_keys(r[0]))} tensors)')
    all_ok &= s3_s2

    if args.saved_exp:
        for ep in args.epochs:
            st = ckpt_state(args.saved_exp, ep)
            all_ok &= compare(f'{args.saved_exp} saved ep{ep}', st, refs)
    else:
        ec = util.gen_exp_config(Args(args.exp_config))
        mc = util.gen_model_config(ec)
        net = getattr(models, ec.model.name)(mc)
        init = {k: v.detach().clone() for k, v in net.state_dict().items() if k.startswith(PREFIXES)}
        util.load_pretrain(net, ec)  # exactly what train.py does
        after = net.state_dict()
        changed = sum(1 for k in init if not torch.equal(init[k], after[k]))
        print(f'random-init -> after load_pretrain: {changed}/{len(init)} encoder tensors changed '
              f'(must be all, else the match below could be vacuous)')
        all_ok &= (changed == len(init))
        all_ok &= compare(f'{args.exp_config} after load_pretrain (model {ec.model.name})', after, refs)

    print('\nRESULT:', 'PASS' if all_ok else 'FAIL')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
