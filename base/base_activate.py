import numpy as np

class Activate:
    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        s = Activate.sigmoid(x)
        return s*(1-s)

    @staticmethod
    def tanh(x):
        return 2/(1 + np.exp(-2*x)) - 1

    @staticmethod
    def dtanh(x):
        return 1 - Activate.tanh(x)**2
