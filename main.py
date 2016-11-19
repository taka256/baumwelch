import numpy as np
from matplotlib import pyplot as plt
import sys
from model.baumwelch import  BaumWelch
from model.dice import Dice

if __name__ == '__main__':

    argv = sys.argv

    # true parameters
    A = np.array([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
    B = np.array([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
    rho = np.array([1.0, 0.0, 0.0])

    # init estimated parameters
    A_e = np.array([[0.15, 0.60, 0.25], [0.25, 0.15, 0.60], [0.60, 0.25, 0.15]])
    B_e = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    rho_e = np.array([1.0, 0.0, 0.0])

    # generate spots of dice
    dice = Dice(A, B, rho)
    s, x = dice.generate_spots(n = 1000)

    # estimate parameters
    baum = BaumWelch(A_e, B_e, rho_e)
    baum.estimate(x, T = 200)

    # save graph of parameters
    baum.graph_A(A)
    baum.graph_B(B)
    baum.graph_P()
