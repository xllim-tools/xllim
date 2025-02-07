import numpy as np

class Bimodal(object):
    """ Bi-modal example.
    F is non-linear and has two antecedents in [0, 1] for each observation: x_obs_1 and x_obs_2 = (1-x_obs_1)
    """

    A = 0.5 * np.array([[1, 2, 2, 1],
                        [0, 0.5, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 3],
                        [0.2, 0, 0, 0],
                        [0, -0.5, 0, 0],
                        [-0.2, 0, -1, 0],
                        [-1, 0, 2, 0],
                        [0, 0, 0, -0.7]]) # injective matrix
    D_dimension = 9
    L_dimension = 4

    def F(self, x):

        Hx = np.copy(x)
        Hx[2] = 4.*(x[2] - 0.5)**2

        Gx = np.exp(Hx)

        y = np.dot(self.A, np.transpose(Gx))
        return y

    def get_D_dimension(self):
        return self.D_dimension

    def get_L_dimension(self):
        return self.L_dimension

    def to_physic(self, x):
        return x

    def from_physic(self, x):
        return x

