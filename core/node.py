import abc
import numpy as np


class BaseComponent:
    """Base component class for neurons and synapses"""
    def __init__(self, name, **kargs):
        """
        initial function
        :param name: name of the component, each component should be different in a circuit
        :param kargs: key arguments
        """
        self.kargs = kargs
        self.name = name
        self.parents = []
        self.children = []
        self.value = None

    def append_children(self, child) -> list:
        """
        append child component in the children list

        :param child: the child component of the current component(neuron or synapse)
        :return: children list
        """
        self.children.append(child)

    def append_parents(self, parent) -> list:
        """
        append parent component in the parents list

        :param parent: the parent component of the current component(neuron or synapse)
        :return: parents list
        """
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


