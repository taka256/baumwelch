import numpy as np
from matplotlib import pyplot as plt
from forward import Forward
from backward import Backward

class BaumWelch(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho
        self.c, self.m = B.shape


    def estimate(self, x, T):
        self.__init_storage(T)
        for t in range(T):
            fw, bw = Forward(self.A, self.B, self.rho), Backward(self.A, self.B, self.rho)
            P, _ = fw.evaluate(x), bw.evaluate(x)
            self.__update(fw.alpha, bw.beta, x)
            self.__store(np.log(P), t)


    def scaled_estimate(self, x, T):
        self.__init_storage(T)
        for t in range(T):
            fw, bw = Forward(self.A, self.B, self.rho), Backward(self.A, self.B, self.rho)
            P, _ = fw.scaled_evaluate(x), bw.scaled_evaluate(x, fw.C)
            self.__scaled_update(fw.alpha, bw.beta, fw.C, x)
            self.__store(np.log(fw.C).sum(), t)


    def graph_A(self, A):
        self.__graph(self.__A, A, 'A')


    def graph_B(self, B):
        self.__graph(self.__B, B, 'B')


    def graph_P(self):
        seq = np.arange(self.__logP.size)
        plt.plot(seq, self.__logP)
        plt.savefig('logP.png')
        plt.clf()


    def __update(self, alpha, beta, x):
        denom = alpha[-1, :].sum().reshape(-1, 1)
        numer = self.__numer(alpha, beta, x)
        gamma = alpha * beta / denom
        xi = numer / denom
        self.__update_param(gamma, xi, x)


    def __scaled_update(self, alpha, beta, C, x):
        denom = np.tile(C[1:], (self.c, 1)).T
        numer = self.__numer(alpha, beta, x)
        gamma = alpha * beta
        xi = numer / denom
        self.__update_param(gamma, xi, x)


    def __numer(self, alpha, beta, x):
        _a = (np.tile(alpha[:-1, :], (self.c, 1, 1)).swapaxes(0, 1) * self.A.T).transpose(2, 0, 1)
        _b = self.B[:, x[1:]].T * beta[1:, :]
        return _a * _b


    def __update_param(self, gamma, xi, x):
        self.A = xi.sum(axis = 1) / gamma[:-1, :].sum(axis = 0).reshape(-1, 1)
        self.B = np.tensordot(gamma, [(x == k) for k in range(self.m)], axes = (0, 1)) / gamma.sum(axis = 0).reshape(-1, 1)
        self.rho = gamma[0, :]


    def __init_storage(self, T):
        self.__logP = np.zeros(T)
        self.__A = np.zeros((T + 1, self.c, self.c))
        self.__B = np.zeros((T + 1, self.c, self.m))
        self.__A[0, :, :] = self.A
        self.__B[0, :, :] = self.B


    def __store(self, logP, t):
        self.__logP[t] = logP
        self.__A[t+1, :, :] = self.A
        self.__B[t+1, :, :] = self.B


    def __graph(self, param_e, param_t, param_str):
        seq = np.tile(np.arange(param_e.shape[0]), (self.c, 1)).T
        for (i, pe) in enumerate(param_e.transpose(2, 0, 1)):
            plt.ylim(0.0, 1.0)
            plt.plot(seq, np.tile(param_t[:, i].reshape(-1, 1), seq.shape[0]).T, 'k-', linewidth = 0.5)
            plt.plot(seq, pe)
            plt.savefig('{}{}.png'.format(param_str, i + 1))
            plt.clf()
