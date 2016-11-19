import numpy as np

class Forward(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho
        self.c = A.shape[0]

    def evaluate(self, x):
        self.alpha = np.zeros((x.size, self.c))
        self.alpha[0, :] = self.rho * self.B[:, x[0]]
        for (i, x_t) in enumerate(x[1:]):
            self.alpha[i+1, :] = self.__forward(x_t, i)
        return self.alpha[-1, :].sum()

    def __forward(self, x_t, i):
        return self.alpha[i, :].dot(self.A) * self.B[:, x_t]
