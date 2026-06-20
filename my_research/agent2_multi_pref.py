import numpy as np
import random

LANDMARK_POSITIONS = {
    'Red':   np.array([-9.0,  9.0]),
    'Green': np.array([-9.0, -9.0]),
    'Blue':  np.array([ 9.0, -9.0]),
    'Cyan':  np.array([ 9.0,  9.0]),
}

LANDMARK_NAMES = ['Red', 'Green', 'Blue', 'Cyan']


class MultiPrefAgent:
    """
    各エピソードの開始時にランダムにランドマークを選択し、
    そのランドマークへ向かって移動するエージェント。

    使い方:
        agent = MultiPrefAgent()
        for episode in range(N):
            pref_name = agent.reset()   # エピソードごとに選好をリセット
            for step in range(T):
                action = agent.get_action(current_pos)
    """

    def __init__(self, seed=None, speed=1.0):
        self.speed = speed
        self.stop_threshold = max(speed, 0.01)  # 速度と同程度の閾値
        self.target_name = None
        self.target_pos = None
        self._rng = random.Random(seed)

    def reset(self):
        """エピソード開始時に呼ぶ。ランダムに選好を選択し、ランドマーク名を返す。"""
        self.target_name = self._rng.choice(LANDMARK_NAMES)
        self.target_pos = LANDMARK_POSITIONS[self.target_name].copy()
        return self.target_name

    def get_action(self, current_pos):
        direction = self.target_pos - current_pos
        distance = np.linalg.norm(direction)
        if distance < self.stop_threshold:
            return np.array([0.0, 0.0])
        return direction / distance * self.speed


class LandmarkFollowerAgent:
    """
    評価用: 特定のランドマークへ向かい続けるエージェント。

    使い方:
        agent = LandmarkFollowerAgent('Green')
        action = agent.get_action(current_pos)
    """

    def __init__(self, name, speed=1.0):
        assert name in LANDMARK_POSITIONS, f"Unknown landmark: {name}"
        self.target_name = name
        self.target_pos = LANDMARK_POSITIONS[name].copy()
        self.speed = speed
        self.stop_threshold = max(speed, 0.01)

    def get_action(self, current_pos):
        direction = self.target_pos - current_pos
        distance = np.linalg.norm(direction)
        if distance < self.stop_threshold:
            return np.array([0.0, 0.0])
        return direction / distance * self.speed
