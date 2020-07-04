import abc
import logging


class Logger:
    def __init__():
        logging.basicConfig(filename="test.log", filemode='w')

    def log(self, msg):
        logging.warning(msg)


class FunctionalModelInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
                hasattr(subclass, 'F') and callable(subclass.F) and
                hasattr(subclass, 'get_D_dimension') and callable(subclass.get_D_dimension) and
                hasattr(subclass, 'get_L_dimension') and callable(subclass.get_L_dimension) and
                hasattr(subclass, 'to_physic') and callable(subclass.to_physic) and
                hasattr(subclass, 'from_physic') and callable(subclass.from_physic) or
                NotImplemented
        )

    @abc.abstractmethod
    def F(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def get_D_dimension(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_L_dimension(self):
        raise NotImplementedError

    @abc.abstractmethod
    def to_physic(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def from_physic(self, x):
        raise NotImplementedError
