import numpy as np

class GreenFollowerAgent:
    """
    緑ランドマーク（左下 -9,-9）に向かって動くエージェント2
    座標追従でシンプルに実装
    """
    def __init__(self):
        self.green_pos = np.array([-9.0, -9.0])
        self.speed = 1.0

    def get_action(self, current_pos):
        """
        現在位置から緑ランドマークへの方向に動く
        戻り値：m_tと同じフォーマット（2次元ベクトル）
        """
        direction = self.green_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < 1.0:
            # 緑に十分近づいたら停止
            return np.array([0.0, 0.0])

        # 方向を正規化してspeedをかける
        action = direction / distance * self.speed
        return action
