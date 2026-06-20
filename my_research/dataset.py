import numpy as np
import torch

def generate_preference_data(num_samples=5000):
    # 3つのターゲット（赤・青・緑）の座標 (x, y) をランダムに生成 [0, 1]
    # v: [赤x, 赤y, 青x, 青y, 緑x, 緑y]
    env_v = np.random.rand(num_samples, 6)
    
    # 各サンプルの「他者の好み」をランダムに決める (0:赤, 1:青, 2:緑)
    # これは学習時の「隠された正解」になります
    other_prefs = np.random.randint(0, 3, size=num_samples)
    
    # 他者の移動ベクトル m2 (好みのターゲットに向かって少し動く)
    m2 = np.zeros((num_samples, 2))
    for i in range(num_samples):
        target_idx = other_prefs[i]
        target_pos = env_v[i, target_idx*2 : target_idx*2+2]
        # ターゲット方向に移動（少しノイズを加える）
        m2[i] = target_pos * 0.1 + np.random.normal(0, 0.01, 2)

    # 自分の移動ベクトル m1 (自分はとりあえずランダムか静止でOK)
    m1 = np.random.normal(0, 0.01, (num_samples, 2))

    # 保存
    np.savez('pref_data.npz', v=env_v, m1=m1, m2=m2, labels=other_prefs)
    print(f"成功: {num_samples}件のデータを 'pref_data.npz' として保存しました。")

if __name__ == "__main__":
    generate_preference_data()