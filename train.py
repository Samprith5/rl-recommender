from environment import UserEnv
from qlearning import choose, update
import numpy as np

env = UserEnv()

for _ in range(1000):
    s = np.random.randint(0,10)
    a = choose(s)
    _, r = env.step(np.random.rand(3))
    s2 = np.random.randint(0,10)
    update(s,a,r,s2)

print("Training done")
