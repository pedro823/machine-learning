import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.base_learning import BaseLearning
from base.base_activate import Activate as act

epoch = 0
side = 40
ratio = 1

class SupervisedNetwork(BaseLearning):
    """
        Supervised Network
        Created by:
        IceMage144 (João Gabriel Basi)
    """
    def __init__(self, numNeurons, activate="tanh"):
        if type(activate) == str:
            f = getattr(act, activate)
            df = getattr(act, "d" + activate)
            self.activate = [[f, df] for i in range(len(numNeurons))]
        elif type(activate) == list:
            self.activate = []
            for e in activate:
                if type(e) == str:
                    f = getattr(act, activate)
                    df = getattr(act, "d" + activate)
                    self.activate.append([f, df])
                elif type(e) == list:
                    self.activate.append(e)
        self.numNeurons = numNeurons
        self.weights = []
        self.bias = []
        for i in range(len(numNeurons) - 1):
            self.weights.append(np.random.rand(numNeurons[i], numNeurons[i+1])*2 - 1)
            self.bias.append(np.random.rand(numNeurons[i+1])*2 - 1)

    def apply(self, inp):
        hVals = np.array(inp)
        for i in range(len(self.weights)):
            hVals = self.activate[i][0](np.dot(hVals, self.weights[i]) + self.bias[i])
        return hVals

    def statistics(self, data):
        error = 0
        for p in data:
            error += (self.apply(p[0]) - p[1])**2
        statistics = {
            "avg_error": error/len(data),
            "error_sum": error,
            "weights": self.weights,
            "biases": self.bias
        }
        return statistics

    def train(self, inp, expected, lr=0.03):
        # Feedforward
        hVals = np.array(inp)
        res = [copy.deepcopy(hVals)]
        for i in range(len(self.weights)):
            hVals = np.dot(hVals, self.weights[i]) + self.bias[i]
            res.append(copy.deepcopy(hVals))
            hVals = self.activate[i][0](hVals)

        # Claculate first error
        error = (expected - hVals)*self.activate[-1][1](res[-1])

        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate gradients
            dw = lr*np.outer(self.activate[i][0](res[i]), error)
            db = lr*error

            # Calculate next error
            error = (self.weights[i]@error)*self.activate[i][1](res[i])

            # Sum gradients
            self.weights[i] += dw
            self.bias[i] += db

    def copy(self):
        return copy.deepcopy(self)

def xor_test():
    nn = SupervisedNetwork([2, 2, 1])
    x = [[[0, 0], [-1]],
         [[0, 1], [1]],
         [[1, 0], [1]],
         [[1, 1], [-1]]]
    fig, ax = plt.subplots(figsize=(5, 5))
    mat = [[[ratio*(2*x/side - 1), ratio*(1 - 2*y/side)] for x in range(side)] for y in range(side)]
    c = ax.imshow([[nn.apply(mat[i][j])[0] for i in range(side)] for j in range(side)],
                  interpolation='nearest',
                  aspect='auto',
                  cmap='RdYlGn')
    epoch_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    def update():
        global epoch
        for i in range(100):
            s = random.sample(x, 1)[0]
            nn.train(s[0], s[1])
        epoch += 100
        yield [[nn.apply(mat[i][j])[0] for i in range(side)] for j in range(side)]
    def plot(update):
        global epoch
        c.set_data(update)
        epoch_text.set_text(f"Epoch = {epoch}")
        return c, epoch_text
    ani = FuncAnimation(fig, plot, update, interval=50)
    plt.ylim(0, side)
    plt.xlim(0, side)
    plt.show()
    print(nn.weights)
    print(nn.bias)
    p = input("Point: ")
    while p != "":
        p = [float(i) for i in p.split()]
        print(p)
        print(nn.apply(p))
        p = input("Point: ")


def test(case):
    if case == "circ":
        from circ_data import DATA
    elif case == "quad":
        from quad_data import DATA
    nn = SupervisedNetwork([5, 2, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    mat = [[[ratio*(2*x/side - 1), ratio*(1 - 2*y/side), (ratio*(2*x/side - 1))**2, (ratio*(1 - 2*y/side))**2, ratio*(1 - 2*y/side)*ratio*(2*x/side - 1)] for x in range(side)] for y in range(side)]
    c = ax.imshow([[nn.apply(mat[i][j])[0] for i in range(side)] for j in range(side)],
                  interpolation='nearest',
                  aspect='auto',
                  cmap='RdYlGn')
    epoch_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    def update():
        global epoch
        for i in range(50):
            nn.train(DATA[epoch%len(DATA)][0], DATA[epoch%len(DATA)][1])
            epoch += 1
        yield [[nn.apply(mat[i][j])[0] for i in range(side)] for j in range(side)]
    def plot(update):
        global epoch
        c.set_data(update)
        epoch_text.set_text(f"Epoch = {epoch}")
        return c, epoch_text
    ani = FuncAnimation(fig, plot, update, interval=50)
    plt.ylim(0, side)
    plt.xlim(0, side)
    plt.show()
    print(nn.weights)
    print(nn.bias)
    p = input("Point: ")
    while p != "":
        p = [float(i) for i in p.split()]
        print(p)
        print(nn.apply(p))
        p = input("Point: ")

if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] == "xor":
        xor_test()
    elif sys.argv[1] == "quad":
        test("quad")
    elif sys.argv[1] == "circ":
        test("circ")
