import numpy as np

class Forward(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho
        self.c = A.shape[0]

    def initialize(self, x_0):
        self.alpha = (self.rho * self.B[:, x_0]).reshape(1, -1)

    def evaluate(self, x):
        self.initialize(x[0])
        for x_t in x[1:]:
            self.alpha = np.vstack((self.alpha, self.__forward(x_t)))
        return self.alpha[-1, :].sum()

    def __forward(self, x_t):
        return self.alpha[-1, :].dot(self.A) * self.B[:, x_t]
