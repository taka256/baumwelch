import numpy as np

class Backward(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho
        self.c = A.shape[0]

    def evaluate(self, x):
        self.beta = np.zeros((x.size, self.c))
        self.beta[-1, :] = np.ones(self.c)
        for (i, x_t) in list(enumerate(x[1:]))[::-1]:
            self.beta[i, :] = self.__backward(x_t, i + 1)
        return (self.rho * self.B[:, x[0]] * self.beta[0, :]).sum()

    def __backward(self, x_t, i):
        return (self.A * self.B[:, x_t] * self.beta[i, :]).sum(axis = 1)
