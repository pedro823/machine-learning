import sys
import os
# Adds higher directory to python modules path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import random
from perceptron import Perceptron

class PerceptronWithPocket(Perceptron):
    """
        Perceptron with pocket.
        Differently from perceptron which returns
        the last function it created while training,
        The pocket variant stores the best function
        it created and uses it instead.
        Created by:
        ~razgrizone (Pedro Pereira)
    """

    def __init__(self, data, results, weights=None):
        """
            Initializes a Pocket algorithm instance.
            Uses the same parameters in Perceptron.
        """
        super().__init__(data, results, weights)
        self.pocket = list(self.weights)

    def train(self, rounds=10000):
        """
            Trains with data for `rounds` amounts of rounds.
            Every round, checks if the function created is
            better than the one in the pocket.
            If so, swaps the one in the pocket for it.
            Returns: (int) the number of rounds that actually ran.
        """
        for round_number in range(rounds):
            # shuffle its own data
            # so that it doesn't stay in the same numbers
            # self._shuffle()
            for datapoint, answer in zip(self.data, self.results):
                signal = self._apply_function(datapoint) >= 0
                if signal != answer:
                    self._adjust(datapoint, answer)
                    # Adjusted once: check if weights are better than
                    # the pocket function
                    self._compare_pocket()
                    break
            else:
                # no incorrect answer was found: dataset
                # is fully fit with current weights
                return round_number + 1
        return rounds

    def apply(self, datapoint):
        """
            Applies the pocket function to the datapoint.
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
        s = self._apply_pocket(datapoint)
        return 1 if s >= 0 else -1

    def statistics(self):
        """
            All statistics available:
            percentage_hits -> percentage of correct predictions
                               in initial input data
            weights         -> the final weights in the pocket.
        """
        # The only change is to apply pocket instead
        all_data = len(self.data)
        hits = 0
        for data, result in zip(self.data, self.results):
            signal = self._apply_pocket(data) >= 0
            if signal == result:
                hits += 1

        statistics = {
            'percentage_hits': round(hits*100/all_data, 2),
            'weights': self.pocket
        }
        return statistics

    # private, heritable

    def _compare_pocket(self):
        """
            Compares if the new function 'weights' is better than
            the pocket function. if so, replaces it.
        """
        pocket_hits = 0
        weights_hits = 0
        for datapoint, answer in zip(self.data, self.results):
            signal_weights = self._apply_function(datapoint) >= 0
            signal_pocket = self._apply_pocket(datapoint) >= 0
            pocket_hits += int(signal_pocket == answer)
            weights_hits += int(signal_weights == answer)
        if weights_hits > pocket_hits:
            # new function is better than pocket
            self.pocket = list(self.weights)

    def _apply_pocket(self, datapoint):
        """ applies sum(pi * xi) """
        s = self.pocket[0] # 1 * w[0]
        for wi, xi in zip(self.pocket[1:], datapoint):
            s += wi * xi
        return s

if __name__ == '__main__':
    from data import INPUT, RESULTS

    p = PerceptronWithPocket(INPUT, RESULTS)
    amount_rounds = p.train(30000)
    print(p.statistics(), 'rounds trained =', amount_rounds)
