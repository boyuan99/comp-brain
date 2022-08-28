import abc
import numpy as np
from collections import OrderedDict


class BaseComponent:

    def __init__(self, name, **kargs):
        self.kargs = kargs
        self.name = name
        self.parents = []
        self.children = []
        self.value = None

    def append_children(self, child):
        self.children.append(child)

    def append_parents(self, parent):
        self.parents.append(parent)


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


