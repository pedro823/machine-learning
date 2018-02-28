import sys
import os
# Adds higher directory to python modules path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import random
from base.base_learning import BaseLearning

class Perceptron(BaseLearning):
    """
        Perceptron
        Created by:
        ~razgrizone (Pedro Pereira)
    """
    class PerceptronError(Exception):
        pass

    def __init__(self, data, results):
        """
            Initializes a perceptron algorithm instance.
            data (list|tuple)    -> list containing points in nth-dimension.
                                    example: [(0.3, 0.5, 3), (4.3, 4.9, 1)]
            results (list|tuple) -> list containing the answers for the data.
                                    example: [True, True, False]
            constraint: len(data) == len(results)
            constraint: len(data[0]) == len(data[1]) == len(data[2])...
        """
        if type(data) is not list and type(data) is not tuple:
            raise PerceptronError('data must be a list or a tuple')
        if len(data) < 1:
            raise PerceptronError('data must not be empty')
        if len(data) != len(results):
            raise PerceptronError('data and results must be of equal length')

        self.data = list(data)
        self.results = results
        self.dimensions = len(data[0]) + 1
        for i in data:
            if len(i) != self.dimensions - 1:
                raise PerceptronError('data is not consistent in dimensions')
        self.weights = [random.random() * 10 for i in range(self.dimensions)]

    def train(self, rounds=10000):
        """
            Trains with data for `rounds` amount of rounds.
        """
        for round_number in range(rounds):
            # so that it doesn't stay in the same numbers
            random.shuffle(self.data)
            for datapoint, answer in zip(self.data, self.results):
                signal = self.__apply_function(datapoint) >= 0
                if signal != answer:
                    self.__adjust(datapoint, answer)
                    # Adjusted once, go to next round
                    break
            else:
                # no incorrect answer was found: dataset
                # is fully fit with current weights
                break

    def apply(self, datapoint):
        """
            Applies perceptron to new datapoint.
            datapoint (list) -> one point of n dimensions
                                (the same n we trained in).
            returns:
                prediction (-1 or 1)
        """
        s = self.__apply_function(datapoint)
        return 1 if s >= 0 else -1

    def statistics(self):
        """
            All statistics available:
            percentage_hits -> percentage of correct predictions
                               in initial input data
        """
        all_data = len(self.data)
        hits = 0
        for data, result in zip(self.data, self.results):
            signal = self.__apply_function(data) >= 0
            if signal == result:
                hits += 1

        statistics = {
            'percentage_hits': round(hits*100/all_data, 2)
        }
        return statistics

    # private

    def __apply_function(self, datapoint):
        """ applies sum(wi * xi) """
        s = self.weights[0] # 1 * w[0]
        for wi, xi in zip(self.weights[1:], datapoint):
            s += wi * xi
        return s

    def __adjust(self, datapoint, answer):
        """ adjusts datapoint like w <- w + xi * yi """
        # w += xi * yi
        adjust = 1 if answer else -1
        self.weights[0] += adjust # treshold
        for i in range(1, len(self.weights)):
            self.weights[i] += datapoint[i - 1] * adjust


if __name__ == '__main__':
    from data import INPUT, RESULTS

    p = Perceptron(INPUT, RESULTS)
    p.train(30000)
    print(p.statistics())
