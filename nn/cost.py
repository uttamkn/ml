import numpy as np
from numpy.typing import NDArray
from activation import Activation


class CostFunction:
    """Base class for cost functions."""

    def fn(self, output_activations: NDArray, y: NDArray) -> float:
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
        # Add epsilon to avoid log(0)
        eps = 1e-10
        return -np.sum(y * np.log(output_activations + eps))

    def delta(self, output_activations, y, z, activation):
        # For sigmoid or softmax outputs: simplified gradient
        return output_activations - y
