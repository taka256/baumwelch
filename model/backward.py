import numpy as np

class Backward(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho
        self.c = A.shape[0]

    def initialize(self):
        self.beta = np.ones((1, self.c))

    def evaluate(self, x):
        self.initialize()
        for x_t in x[:0:-1]:
            self.beta = np.vstack((self.__backward(x_t), self.beta))
        return (self.rho * self.B[:, x[0]] * self.beta[0, :]).sum()

    def __backward(self, x_t):
        return (self.A * self.B[:, x_t] * self.beta[0, :]).sum(axis = 1)
