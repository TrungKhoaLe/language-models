import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    h = 1 / (1 + np.exp(-z))
    return h
