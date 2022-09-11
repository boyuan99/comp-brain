from compbrain.core.node import BaseComponent
import numpy as np
from collections import OrderedDict
from compbrain.core import CompBrainModelError


class InjectCurrent(BaseComponent):
    """
    the external current injection input as a synapse model

    :argument
        name: the name of the synapse, notice that in a circuit all of the
            synapse's names should be different
        t: a numpy array time series
        presynaptic: the name of the presynaptic neuron, doesn't need to be instantiated
        postsynaptic: the name of the postsynaptic neuron, doesn't need to be instantiated
        kwargs: keyword arguments that overwrite initial conditions of state
            variables and values of parameters

            :current
                a time series containing the current array
            :type
                input current type, needed is no current defined here. The class will generate an
                input current for the model
    """

    def __init__(self, name, presynaptic, postsynaptic, **kwargs):
        super(InjectCurrent, self).__init__(name, **kwargs)

        if 'current' in kwargs.keys():
            self.current = kwargs['current']
        else:
            self.current = []

        if 't' in kwargs.keys():
            self.t = kwargs['t']
        else:
            self.t = np.arange(0, 1, 1e-4)

        if 'type' in kwargs.keys():
            if kwargs['type'] == 'step':
                if 'intensity' in kwargs.keys():
                    self.current = np.zeros_like(self.t)
                    self.current[int(0.2 * len(self.t)):int(0.7 * len(self.t))] = kwargs['intensity']
                else:
                    self.current = np.zeros_like(self.t)
                    self.current[int(0.2 * len(self.t)):int(0.7 * len(self.t))] = 5

            else:
                raise CompBrainModelError("no {} inject type implemented".format(kwargs['type']))

        else:
            if 'intensity' in kwargs.keys():
                self.current = np.zeros_like(self.t)
                self.current[int(0.2 * len(self.t)):int(0.7 * len(self.t))] = kwargs['intensity']
            else:
                self.current = np.zeros_like(self.t)
                self.current[int(0.2 * len(self.t)):int(0.7 * len(self.t))] = 5

        self.count = 0
        self.states: OrderedDict = OrderedDict(I_ext=[], I_syn=[])
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic

    def get_V_pre(self):
        """
        get the presynaptic voltage, None for injection current cause no
        presynaptic neuron exists and not needed for the computation process
        :return: None
        """
        return None

    def get_V_post(self):
        """
        get the presynaptic voltage, None for injection current cause not
        needed for the computation process
        :return: None
        """
        return None

    def compute(self, V_pre, V_post):
        f"""
        Injection current function
        query the current value from the generated input current

        :param V_pre: None
        :param V_post: None
        :return: dict(I_ext, I_syn)
        """
        I_ext = self.current[self.count]
        self.states['I_ext'].append(self.current[self.count])
        self.count += 1
        return self.states
