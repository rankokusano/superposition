import json, statistics as stt
E='data/result/v11/stage2'
eps=[160,170,180,190,200]
mean=lambda xs: stt.mean(xs)
for st in ('green','pref'):
    for c in ('random','mixed'):
        J=[json.load(open(f'{E}/eval/{st}_{c}_s3_ep{e}.json')) for e in eps]
        T=[json.load(open(f'{E}/t1t3/{st}_{c}_s3_ep{e}.json')) for e in eps]
        F=[json.load(open(f'{E}/t4t6/{st}_{c}_s3_ep{e}.json')) for e in eps]
        M=[json.load(open(f'{E}/m1d_other/{st}_{c}_s3_ep{e}.json')) for e in eps]
        print(f"{st} {c}: corr green_set {mean([j['corr_Q2_PA2']['green_set']['all'] for j in J]):+.3f} orig {mean([j['corr_Q2_PA2']['green_set']['original'] for j in J]):+.3f} swap {mean([j['corr_Q2_PA2']['green_set']['swap_rg'] for j in J]):+.3f} pref {mean([j['corr_Q2_PA2']['pref_test']['all'] for j in J]):+.3f}"
              f" | A1step {mean([j['motion_readout']['green_set']['Q2_to_A1_step_dir'] for j in J]):.3f} A2step {mean([j['motion_readout']['green_set']['Q2_to_A2_step_dir'] for j in J]):.3f}"
              f" | selfL1 {mean([j['self_vision_L1_per_pixel']['green'] for j in J]):.3f}"
              f" | T3 self g/z/d6 {mean([t['green']['gen']['self_L1'] for t in T]):.3f}/{mean([t['green']['zero']['self_L1'] for t in T]):.3f}/{mean([t['green']['d6']['self_L1'] for t in T]):.3f} fp {mean([t['green']['gen']['other_fp_mse'] for t in T]):.3f}/{mean([t['green']['zero']['other_fp_mse'] for t in T]):.3f}/{mean([t['green']['d6']['other_fp_mse'] for t in T]):.3f}"
              f" | T1 {mean([t['green']['T1_A2pos']['all'] for t in T]):.3f} var2-5 {mean([t['green']['spread']['var_share_bins2_5'] for t in T]):.3f} std {mean([t['green']['spread']['std_all'] for t in T]):.3f}"
              f" | trk {mean([f['gen']['tracking']['green']['median'] for f in F]):.2f} T4 {mean([f['gen']['T4']['swap_minus_noswap'] for f in F]):+.3f}"
              f" | M1dO g {mean([m['green_gen']['c_cf_other'] for m in M]):+.4f} r {mean([m['red_gen']['c_cf_other'] for m in M]):+.4f}")
