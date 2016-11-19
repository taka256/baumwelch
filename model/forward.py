import numpy as np

class Forward(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho
        self.c = A.shape[0]


    def evaluate(self, x):
        self.__initialize(x)
        for (i, x_t) in enumerate(x[1:]):
            self.alpha[i+1, :] = self.__forward(x_t, i)
        return self.alpha[-1, :].sum()


    def scaled_evaluate(self, x):
        self.__scaled_initialize(x)
        for (i, x_t) in enumerate(x[1:]):
            self.alpha[i+1, :] = self.__scaled_forward(x_t, i)
        return 1.0 / self.C[:].prod()


    def __initialize(self, x):
        self.alpha = np.zeros((x.size, self.c))
        self.alpha[0, :] = self.rho * self.B[:, x[0]]


    def __scaled_initialize(self, x):
        self.__initialize(x)
        self.C = np.zeros(x.size)
        self.C[0] = 1.0 / self.alpha[0, :].sum()
        self.alpha[0, :] *= self.C[0]


    def __forward(self, x_t, i):
        return self.alpha[i, :].dot(self.A) * self.B[:, x_t]


    def __scaled_forward(self, x_t, i):
        _alpha = self.__forward(x_t, i)
        self.C[i + 1] = 1.0 / _alpha.sum()
        return _alpha * self.C[i + 1]
