import numpy as np


def spec(n):
    t = np.linspace(-510, 510, n)
    return np.round(np.clip(np.stack([-t, 510-np.abs(t), t], axis=1), 0, 255)).astype(np.uint8)
