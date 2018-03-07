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
        """ Common class for errors thrown by perceptron. """
        pass

    def __init__(self, data, results, weights=None):
        """
            Initializes a perceptron algorithm instance.
            data (list|tuple)    -> list containing points in nth-dimension.
                                    example: [(0.3, 0.5, 3), (4.3, 4.9, 1)]
            results (list|tuple) -> list of booleans containing the answers
                                    for the data.
                                    example: [True, True, False]
            weights (list)       -> initializes weights as set, instead of
                                    random numbers.
            constraint: len(data) == len(results)
            constraint: len(data[0]) == len(data[1]) == len(data[2])...
            constraint: len(data[0]) + 1 == len(weights)
        """
        if type(data) not in (list, tuple):
            raise PerceptronError('data must be a list or a tuple')
        if len(data) < 1:
            raise PerceptronError('data must not be empty')
        if len(data) != len(results):
            raise PerceptronError('data and results must be of equal length')

        self.data = list(data)
        # For now a constant
        self.shuffle_amount = 500
        self.results = list(results)
        self.dimensions = len(data[0]) + 1
        for i in data:
            if len(i) != self.dimensions - 1:
                raise PerceptronError('data is not consistent in dimensions')
        if weights:
            if len(weights) != self.dimensions:
                raise PerceptronError('Weights do not have the correct '
                                      'dimension. expected'
                                      + str(self.dimensions))
            self.weights = weights
        else:
            self.weights = [random.random() * 10 for i in range(self.dimensions)]

    def train(self, rounds=10000):
        """
            Trains with data for `rounds` amount of rounds.
            (if it is fully fit before `rounds` rounds, it stops).
            Returns: (int) the number of rounds that actually ran.
        """
        for round_number in range(rounds):
            # shuffle its own data
            # so that it doesn't stay in the same numbers
            self._shuffle()
            for datapoint, answer in zip(self.data, self.results):
                signal = self._apply_function(datapoint) >= 0
                if signal != answer:
                    self._adjust(datapoint, answer)
                    # Adjusted once, go to next round
                    break
            else:
                # no incorrect answer was found: dataset
                # is fully fit with current weights
                return round_number + 1
        return rounds

    def apply(self, datapoint):
        """
            Applies perceptron to new datapoint.
            datapoint (list) -> one point of n dimensions
                                (the same n we trained in).
            returns:
                prediction (-1 or 1)
        """
        if type(datapoint) not in (list, tuple):
            raise PerceptronError('datapoint is not a list or a tuple')
        if len(datapoint) != self.dimensions - 1:
            raise PerceptronError('datapoint does not have the correct'
                                  'dimension. Expected', self.dimensions - 1)
        s = self._apply_function(datapoint)
        return 1 if s >= 0 else -1

    def statistics(self):
        """
            All statistics available:
            percentage_hits -> percentage of correct predictions
                               in initial input data
            weights         -> the final weights calculated in
                               perceptron algorithm.
        """
        all_data = len(self.data)
        hits = 0
        for data, result in zip(self.data, self.results):
            signal = self._apply_function(data) >= 0
            if signal == result:
                hits += 1

        statistics = {
            'percentage_hits': round(hits*100/all_data, 2),
            'weights': self.weights
        }
        return statistics

    # private, heritable

    def _apply_function(self, datapoint):
        """ applies sum(wi * xi) """
        s = self.weights[0] # 1 * w[0]
        for wi, xi in zip(self.weights[1:], datapoint):
            s += wi * xi
        return s

    def _adjust(self, datapoint, answer):
        """ adjusts datapoint like w <- w + xi * yi """
        # w += xi * yi
        adjust = 1 if answer else -1
        self.weights[0] += adjust * 1 # treshold
        for i in range(1, len(self.weights)):
            self.weights[i] += datapoint[i - 1] * adjust

    def _shuffle(self):
        """ Shuffles internal dataset """
        # shuffles self.shuffle times
        for i in range(self.shuffle_amount):
            # finds two and swap
            a = random.randint(0, len(self.data) - 1)
            b = random.randint(0, len(self.data) - 1)
            self.data[a], self.data[b] = self.data[b], self.data[a]
            self.results[a], self.results[b] = self.results[b], self.results[a]

if __name__ == '__main__':
    from data import INPUT, RESULTS

    p = Perceptron(INPUT, RESULTS)
    amount_rounds = p.train(30000)
    print(p.statistics(), 'rounds trained =', amount_rounds)
