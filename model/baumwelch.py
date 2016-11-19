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
            # init instances of foward and backward
            fw, bw = Forward(self.A, self.B, self.rho), Backward(self.A, self.B, self.rho)

            # evaluate
            P, _ = fw.evaluate(x), bw.evaluate(x)

            # update parameters
            self.__update(fw, bw, x)

            # store parameters
            self.__store(P, t)


    def graph_A(self, A):
        self.__graph(self.__A, A, 'A')


    def graph_B(self, B):
        self.__graph(self.__B, B, 'B')


    def graph_P(self):
        seq = np.arange(self.__P.size)
        plt.plot(seq, self.__P)
        plt.savefig('logP.png')
        plt.clf()


    def __update(self, fw, bw, x):
        mul = fw.alpha * bw.beta
        mul_sum = mul.sum(axis = 0, keepdims = True).T
        self.A = ((np.tile(fw.alpha[:-1, :], (self.c, 1, 1)).swapaxes(0, 1) * self.A.T).transpose(2, 0, 1) * self.B[:, x[1:]].T * bw.beta[1:, :]).sum(axis = 1) / (mul_sum - mul[-1, :].reshape(-1, 1))
        self.B = np.tensordot(mul, [(x == k) for k in range(self.m)], axes = (0, 1)) / mul_sum
        self.rho = mul[0, :] / fw.alpha[-1, :].sum()


    def __init_storage(self, T):
        self.__P = np.zeros(T)
        self.__A = np.zeros((T + 1, self.c, self.c))
        self.__B = np.zeros((T + 1, self.c, self.m))
        self.__A[0, :, :] = self.A
        self.__B[0, :, :] = self.B


    def __store(self, P, t):
        self.__P[t] = np.log(P)
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
