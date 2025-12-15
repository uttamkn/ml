import numpy as np
from numpy.typing import NDArray
from nn.activation import Activation


class CostFunction:
    """Base class for cost functions."""

    def fn(self, output_activations: NDArray, y: NDArray):
        raise NotImplementedError

    def delta(
        self,
        output_activations: NDArray,
        y: NDArray,
        z: NDArray,
        activation: Activation,
    ) -> NDArray:
        """Return δ = ∂C/∂z for the output layer."""
        raise NotImplementedError


class MeanSquaredError(CostFunction):
    def fn(self, output_activations, y):
        return float(0.5 * np.linalg.norm(output_activations - y) ** 2)

    def delta(self, output_activations, y, z, activation):
        return (output_activations - y) * activation.prime(z)


class CrossEntropy(CostFunction):
    def fn(self, output_activations, y):
        """
        Supports both:
        - binary BCE: y in {0,1}, a = sigmoid
        - categorical CE: y is one-hot, a = softmax
        """
        eps = 1e-12
        a = np.clip(output_activations, eps, 1 - eps)

        # Case 1: Binary classification (scalar output)
        if a.size == 1:
            return float(-(y * np.log(a) + (1 - y) * np.log(1 - a)))

        # Case 2: Multi-class classification (one-hot y)
        return float(-np.sum(y * np.log(a)))

    def delta(self, output_activations, y, z, activation):
        """
        For sigmoid+CE and softmax+CE:
            derivative simplifies to a - y
        """
        return output_activations - y
