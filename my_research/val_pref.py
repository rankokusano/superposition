import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model import SuperpositionNet

# 1. モデルとデータの読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SuperpositionNet().to(device)
model.load_state_dict(torch.load('superposition_model.pth'))
model.eval()

data = np.load('pref_data.npz')
v = torch.FloatTensor(data['v']).to(device)
m1 = torch.FloatTensor(data['m1']).to(device)
m2 = torch.FloatTensor(data['m2']).to(device)
labels = data['labels'] # 正解の好み (0:赤, 1:青, 2:緑)

# 2. プロセス2（他者経路）の隠れ状態 h2 を抽出
print("内部状態を抽出中...")
with torch.no_grad():
    # 順伝播させて、隠れ状態 h2 を取得
    _, _, h2 = model(v, m1, m2)
    # h2[0] (hidden state) を取り出し、numpy配列へ (形状: [サンプル数, hidden_dim])
    h2_states = h2[0].squeeze(0).cpu().numpy()

# 3. PCA（主成分分析）で2次元に圧縮して可視化
# 論文の図4cと同様の手法です
pca = PCA(n_components=2)
h2_pca = pca.fit_transform(h2_states)

# 4. グラフ作成
plt.figure(figsize=(8, 6))
scatter = plt.scatter(h2_pca[:, 0], h2_pca[:, 1], c=labels, cmap='jet', alpha=0.5)
plt.colorbar(scatter, label='Preference (0:Red, 1:Blue, 2:Green)')
plt.title("Visualization of Other's Preference in Shared Module (h2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig('preference_visualization.png')
print("結果を 'preference_visualization.png' に保存しました。")