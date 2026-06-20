import sys
sys.path.append('/work')
import numpy as np
from simulation import creator
from simulation.util import load_config

config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
env = creator.create_environment(config.environment)
env.init()
env.off_display()
env.reset()

# 画像を取得
v, _, _, _, _ = env.step()

print(f"画像サイズ: {v.shape}")
print(f"赤チャンネルの最大値: {v[:,:,0].max()}")
print(f"緑チャンネルの最大値: {v[:,:,1].max()}")
print(f"青チャンネルの最大値: {v[:,:,2].max()}")

# 各ピクセルのRGB値で赤っぽいものを探す
r = v[:,:,0]
g = v[:,:,1]
b = v[:,:,2]

print("\n赤っぽいピクセル（R>150, G<100, B<100）:")
red_pixels = np.where((r > 150) & (g < 100) & (b < 100))
if len(red_pixels[0]) > 0:
    for i in range(min(5, len(red_pixels[0]))):
        row, col = red_pixels[0][i], red_pixels[1][i]
        print(f"  位置({row},{col}): R={r[row,col]}, G={g[row,col]}, B={b[row,col]}")
else:
    print("  見つかりませんでした")
    print("\n全ピクセルのRGB最大値トップ5:")
    flat_r = r.flatten()
    flat_g = g.flatten()
    flat_b = b.flatten()
    for i in np.argsort(flat_r)[-5:][::-1]:
        print(f"  R={flat_r[i]}, G={flat_g[i]}, B={flat_b[i]}")
