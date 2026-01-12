import numpy as np

class UserEnv:
    def __init__(self):
        self.state = np.random.rand(3)

    def step(self, action):
        reward = np.dot(self.state, action)
        self.state = np.random.rand(3)
        return self.state, reward
