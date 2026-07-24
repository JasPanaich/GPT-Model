# Activation functions: gives neural network ability to learn complex patterns
import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]: # Squishes input to 0 - 1 (for probability/binary classification)
        return np.round(1 / (1 + np.exp(-z)), 5)
        pass

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0, z)
        pass
