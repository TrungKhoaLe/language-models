from typing import Tuple
from .activations import sigmoid
import numpy as np


def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, 
                     alpha: float, n_iters: int) -> Tuple[float, np.ndarray]:
    m = X.shape[0]

    for i in range(n_iters):
        # get Z
        Z = X @ theta
        # get the sigmoid of Z
        h = sigmoid(Z)
        # calculate the cost function
        J = -1/m * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
        # update the weights of the theta
        theta = theta - alpha/m * (X.T @ (h - y))

    return float(J), theta
