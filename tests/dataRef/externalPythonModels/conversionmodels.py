"""Small external Python models used by tests/pythonTests/external_model_test.py.

They exercise the conversions between the C++ wrapper (ExternalPythonModel) and Python in situations not
covered by the photometric models of this directory (see issue #67).
"""

import numpy as np


class AffineModelL20(object):
    """Model with L = 20 parameters: Armadillo stores vectors of more than 16 elements on the heap.

    toPhysic / fromPhysic modify their argument in place (and return it), like the models of the documentation.
    """

    L_dimension = 20
    D_dimension = 20
    scalingCoeffs = np.linspace(1.0, 2.9, L_dimension)
    offset = np.linspace(-1.0, 0.9, L_dimension)

    def F(self, x):
        x = self.toPhysic(x.copy())
        return np.cumsum(x**2)

    def getDimensionY(self):
        return self.D_dimension

    def getDimensionX(self):
        return self.L_dimension

    def toPhysic(self, x):
        x[:] = x * self.scalingCoeffs + self.offset
        return x

    def fromPhysic(self, x):
        x[:] = (x - self.offset) / self.scalingCoeffs
        return x


class ReturnStyleModel(object):
    """Model whose toPhysic / fromPhysic return a NEW array and leave their argument untouched."""

    scalingCoeffs = np.array([1.0, 1.2, 2.0])
    offset = np.array([0.0, 0.0, -1.0])

    def F(self, x):
        p = self.toPhysic(x)
        return np.array([p.sum(), p.prod(), p[0] - p[2], 1.0])

    def getDimensionY(self):
        return 4

    def getDimensionX(self):
        return 3

    def toPhysic(self, x):
        return x * self.scalingCoeffs + self.offset

    def fromPhysic(self, x):
        return (x - self.offset) / self.scalingCoeffs


class InvalidModel(object):
    """Invalid model: wrong output sizes, and F returns None."""

    def F(self, x):
        return None

    def getDimensionY(self):
        return 3

    def getDimensionX(self):
        return 3

    def toPhysic(self, x):
        return np.zeros(2)

    def fromPhysic(self, x):
        return [1.0, 2.0, 3.0, 4.0]
