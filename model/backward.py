import numpy as np

class Backward(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho
        self.c = A.shape[0]


    def evaluate(self, x):
        self.__initialize(x)
        self.__backward_seq(x, self.__backward)
        return (self.rho * self.B[:, x[0]] * self.beta[0, :]).sum()


    def scaled_evaluate(self, x, C):
        self.__scaled_initialize(x, C)
        self.__backward_seq(x, self.__scaled_backward)
        return self.C.prod()


    def __initialize(self, x):
        self.beta = np.zeros((x.size, self.c))
        self.beta[-1, :] = np.ones(self.c)


    def __scaled_initialize(self, x, C):
        self.__initialize(x)
        self.C = C
        self.beta[-1, :] /= self.C[-1]


    def __backward_seq(self, x, backward_func):
        for (i, x_t) in list(enumerate(x[1:]))[::-1]:
            self.beta[i, :] = backward_func(x_t, i + 1)


    def __backward(self, x_t, i):
        return (self.A * self.B[:, x_t] * self.beta[i, :]).sum(axis = 1)


    def __scaled_backward(self, x_t, i):
        _beta = self.__backward(x_t, i)
        return _beta / self.C[i]
