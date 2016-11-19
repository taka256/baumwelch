import numpy as np

class Dice(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho


    def generate_spots(self, n):
        s, x = [], []
        s += self.__spots(self.rho.reshape(1, -1), 0)
        for n in range(n):
            x += self.__spots(self.B, s[-1])
            s += self.__spots(self.A, s[-1])

        return np.array(s[:-1]), np.array(x)


    def __spots(self, theta, s_t):
        p = np.random.rand()
        for i in range(theta.shape[0]):
            if p < theta[s_t, :i+1].sum():
                break
        return [i]
