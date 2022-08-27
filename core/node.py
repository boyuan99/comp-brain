import abc
import numpy as np
from collections import OrderedDict
from core import default_circuit


class BaseComponent:
    Default_Params: OrderedDict = OrderedDict()
    """A dictionary of (name, value) pair for parameters"""

    Default_States: OrderedDict = OrderedDict()
    """A dictionary of state variables"""

    def __init__(self, name, **kargs):
        self.kargs = kargs
        self.name = name
        self.value = None
        self.value_array = np.array([])


    @abc.abstractmethod
    def compute(self):
        """
        abstract method for computing the component value
        :return:
        """
    def reset_value(self, recursive:bool =True):
        self.value_array = np.append(self.value_array, self.value)
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()


