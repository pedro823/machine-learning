import numpy as np
# Adds higher directory to python modules path.
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# local inclusions
from base.base_learning import BaseLearning

class LinearRegresionMatrix(BaseLearning):
    """
        Linear Regression (Matrix variant)
        Created By:
        ~razgrizone (Pedro Pereira)
    """


    class LinearRegresionError(Exception):
        """ Common class for errors thrown by LinearRegression. """
        pass

    def __init__(self, data, results):
        """
            Initializes linear regression model with training data set.
            data (list|tuple)    -> list containing points in nth-dimension.
                                    example: [(0.3, 0.5, 3), (4.3, 4.9, 1)]
            results (list|tuple) -> list containing the answers for the data.
                                    example: [0.3, 800.2, 123.4]
            constraint: len(data) == len(results)
            constraint: len(data[0]) == len(data[1]) == len(data[2])...
        """

        self.X = self.build_matrix(data)
        self.data = list(data)
        self.results = list(results)
        self.pseudo_inverse_X = None
        self.plane = None

    def train(self):
        """
            Trains linear regression (which just calculates the
            pseudo-inverse of X)
        """
        self.pseudo_inverse_X = np.linalg.pinv(self.X)
        print(self.pseudo_inverse_X)
        y = np.array([[i] for i in self.results])
        self.plane = (self.pseudo_inverse_X * y).tolist()
        self.plane = [i[0] for i in self.plane]

    def statistics(self):
        """
            All statistics available:
            pseudo_inverse (list[list]) -> The calculated pseudo inverse of
                                           the initial matrix of data
            squared_error  (float)      -> How much does the pseudo_inverse
                                           matrix misses the real data set?
        """
        if self.pseudo_inverse_X is None:
            raise LinearRegresionError('LinearRegression was not trained. '
                                       'Use train() before using statistics()')
        squared_error = 0
        for datapoint, result in zip(self.data, self.results):
            squared_error += (self.apply(datapoint) - result)**2
        statistics = {
            'pseudo_inverse': np.matrix.tolist(self.pseudo_inverse_X),
            'squared_error': squared_error,
            'avg_squared_error': squared_error / len(self.results)
        }
        return statistics

    def apply(self, datapoint):
        """
            Applies linear regression to datapoint.
            datapoint (list) -> A data point with the same dimension of
                                the initial data list.
                                example: [4, 3, 4]
            returns: the prediction of the value of the datapoint.
        """
        # transposed(w) * x
        if self.pseudo_inverse_X is None:
            raise LinearRegresionError('LinearRegression was not trained.'
                                       ' Use train() before using apply()')
        s = self.plane[0]
        s += sum([self.plane[idx + 1] * x for idx, x in enumerate(datapoint)])
        return s

    @classmethod
    def build_matrix(cls, data):
        """
            Builds X based on initial data.
            X is a matrix such that

            X = [
                    [----x1-->],
                    [----x2-->],
                    ...
                    [----xn-->]
                ]
        """
        try:
            return np.matrix([[1] + i for i in data])
        except TypeError as ex:
            raise LinearRegresionError('numpy error: '
                                       + str(ex)
                                       + '\n\tProbably len(xi) != len(xj)'
                                       + 'for some i, j in the dataset')

if __name__ == '__main__':
    from data import INPUT, RESULTS
    l = LinearRegresionMatrix(INPUT, RESULTS)
    l.train()
    print(l.statistics())
