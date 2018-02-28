from abc import ABCMeta, abstractmethod

class BaseLearning(metaclass=ABCMeta):

    @abstractmethod
    def train(self):
        """
            Implements model training.
        """
        pass

    @abstractmethod
    def statistics(self):
        """
            Prints all statistics possible for the model.
        """
        pass

    @abstractmethod
    def apply(self):
        """
            Applies trained model into new dataset.
        """
        pass
