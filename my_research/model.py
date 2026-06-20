import torch
import torch.nn as nn

class SuperpositionNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, move_dim=2):
        super(SuperpositionNet, self).__init__()
        
        # 1. 2つのエンコーダー (論文の Visual Encoder-1, 2 に相当) [cite: 139]
        # 構造は同じだが、重み（パラメータ）は別々に学習される [cite: 475, 531]
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(input_dim, hidden_dim)
        
        # 2. 共有モジュール (論文の Shared Module に相当) [cite: 141]
        # 論文同様、LSTMを使用し、プロセス1と2でこの単一の重みを共有する [cite: 142, 482]
        self.shared_lstm = nn.LSTM(hidden_dim + move_dim, hidden_dim, batch_first=True)
        
        # 3. 予測器 (論文の Visual Predictor に相当) [cite: 148]
        # 2つのプロセス出力を統合して未来を予測する [cite: 485]
        self.predictor = nn.Linear(hidden_dim * 2, move_dim)

    def forward(self, v, m1, m2, h1=None, h2=None):
        # プロセス1 (自己を想定) [cite: 138]
        f_v1 = torch.relu(self.enc1(v))
        input1 = torch.cat([f_v1, m1], dim=-1).unsqueeze(1) # [batch, seq=1, dim]
        out1, h1 = self.shared_lstm(input1, h1)
        
        # プロセス2 (他者を想定) [cite: 138]
        f_v2 = torch.relu(self.enc2(v))
        input2 = torch.cat([f_v2, m2], dim=-1).unsqueeze(1)
        out2, h2 = self.shared_lstm(input2, h2)
        
        # 論文の数式(5)のように、両出力を統合して次の動きを予測 [cite: 466, 485]
        combined = torch.cat([out1.squeeze(1), out2.squeeze(1)], dim=-1)
        prediction = self.predictor(combined)
        
        return prediction, h1, h2