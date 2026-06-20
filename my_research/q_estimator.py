import torch
import torch.nn as nn

class QEstimator(nn.Module):
    """
    教授指示 Step 2：
    シーンベクトルs（飯塚さんのネットワークのos）と
    m_t（他者の運動）からQ値を推測するネットワーク

    入力：
        s：他者の内部状態（128次元）
        m_t：他者の運動（2次元）
    出力：
        Q値（実数1つ）
    """
    def __init__(self, scene_dim=128, motion_dim=2, hidden_dim=64):
        super(QEstimator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(scene_dim + motion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s, m):
        """
        s: [batch, 128]
        m: [batch, 2]
        """
        x = torch.cat([s, m], dim=-1)
        q = self.net(x)
        return q
