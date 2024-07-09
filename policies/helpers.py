import numpy as np

def random_argmax(x):
    return np.random.choice(np.flatnonzero(x == x.max()))