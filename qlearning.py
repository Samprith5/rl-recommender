import numpy as np

Q = np.zeros((10,10))
lr, gamma = 0.1, 0.9

def choose(state):
    return np.argmax(Q[state])

def update(s,a,r,s2):
    Q[s,a] += lr * (r + gamma * np.max(Q[s2]) - Q[s,a])
