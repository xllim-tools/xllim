import math

import numpy as np


class TestModel(object):
    def F(self, x):
        A = np.array([[1, 2, 2, 1],
                      [0, 0.5, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 3],
                      [0.2, 0, 0, 0],
                      [0, -0.5, 0, 0],
                      [-0.2, 0, -1, 0],
                      [-1, 0, 2, 0],
                      [0, 0, 0, -0.7]])
        A *= 0.5
        Hx = np.ones(self.get_L_dimension())
        Gx = np.ones(self.get_L_dimension())
        Hx[0] = x[0]
        Hx[1] = x[1]
        Hx[2] = 4. * math.pow(x[2] - 0.5, 2)
        Hx[3] = x[3]

        Gx[0] = math.exp(Hx[0])
        Gx[1] = math.exp(Hx[1])
        Gx[2] = math.exp(Hx[2])
        Gx[3] = math.exp(Hx[3])

        y = np.dot(A, Gx.transpose())
        return y

    def get_D_dimension(self):
        return 9

    def get_L_dimension(self):
        return 4

    def to_physic(self, x):
        return x

    def from_physic(self, x):
        return x

