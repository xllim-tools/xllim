import numpy as np

class Unimodal(object):
    """ Uni modal example
    Non-linear and injective function 
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
        x= np.array(x)
        x_exp= np.exp(x) # exponential part
        y = np.dot(self.A, x_exp) # linear part
        for d in range(4): # quadratic part
            y[d] += 5*(x_exp[d] + 0.5)**2
        for d in range(4,8):
            y[d] += 5*(x_exp[d-4] + 0.5)**2
        return y

    def get_D_dimension(self):
        return self.D_dimension

    def get_L_dimension(self):
        return self.L_dimension

    def to_physic(self, x):
        return x

    def from_physic(self, x):
        return x
