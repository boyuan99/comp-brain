import abc
import numpy as np
from collections import OrderedDict
from .circuit import Circuit, default_circuit


class BaseComponent:
    Default_Params: OrderedDict = OrderedDict()
    """A dictionary of (name, value) pair for parameters"""

    Default_States: OrderedDict = OrderedDict()
    """A dictionary of state variables"""

    def __init__(self, *parents, **kargs):
        self.kargs = kargs
        self.circuit = kargs.get("circuit", default=default_circuit)
        self.need_save = kargs.get("need_save", True)
        self.gen_node_name(**kargs)
        self.parents = list(parents)
        self.children = []
        self.value = None
        self.value_array = np.array([])

        for parent in parents:
            parent.children.append(self)

        self.circuit.add_component(self)


    def get_parents(self):
        return self.parents


    def get_children(self):
        return self.children


    def flow(self):
        for parent in self.parents:
            if parent.value is None:
                parent.flow()

        self.compute()


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


    def gen_node_name(self, **kwargs):
        self.name = kwargs.get("name", "{}:{}".format(self.__class__.__name__,
                                                      self.circuit.node_count()))

        if self.circuit.name_scope:
            self.name = "{}/{}".format(self.circuit.name_scope, self.name)

