import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import SuperpositionNet

# 1. データの読み込み
data = np.load('pref_data.npz')
v = torch.FloatTensor(data['v'])   # 環境（色の配置）
m1 = torch.FloatTensor(data['m1']) # 自己の動き
m2 = torch.FloatTensor(data['m2']) # 他者の動き

# 2. モデルのインスタンス化と設定
# GPUが使える場合はGPUに転送
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SuperpositionNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss() # 予測誤差の計算用

# データをdeviceへ
v, m1, m2 = v.to(device), m1.to(device), m2.to(device)

# 3. 学習ループ
epochs = 500
print(f"学習を開始します ({device} を使用)")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # モデルを実行して未来（今の他者の動き）を予測
    # 論文の予測学習 [cite: 10, 150] の簡易版です
    prediction, _, _ = model(v, m1, m2)
    
    # 実際の動きと予測の差を計算して学習
    loss = criterion(prediction, m2)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# 4. 学習済みモデルの保存
torch.save(model.state_dict(), 'superposition_model.pth')
print("学習完了: 'superposition_model.pth' を保存しました。")