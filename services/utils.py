import numpy as np


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.default_rng(seed=42).permutation(len(a))
    return a[p], b[p]