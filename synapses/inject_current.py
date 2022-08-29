from core.node import BaseComponent
import numpy as np
from collections import OrderedDict


class InjectCurrent(BaseComponent):

    def __init__(self, name, t, presynaptic, postsynaptic, **kwargs):
        super(InjectCurrent, self).__init__(name, **kwargs)

        if 'current' in kwargs.keys():
            self.current = kwargs['current']
        else:
            self.current = []

        if 'type' in kwargs.keys():
            if kwargs['type'] == 'step':
                self.current = np.zeros_like(t)
                self.current[int(0.2 * len(t)):int(0.7 * len(t))] = 5

            else:
                raise ValueError("no {} inject type implemented".format(kwargs['type']))

        else:
            self.current = np.zeros_like(t)
            self.current[int(0.2 * len(t)):int(0.7 * len(t))] = 5

        self.count = 0
        self.states: OrderedDict = OrderedDict(I_syn=[])
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic

    def get_V_pre(self):
        return None

    def get_V_post(self):
        return None

    def compute(self, V_pre, V_post):
        I_ext = self.current[self.count]
        self.states['I_syn'].append(self.current[self.count])
        self.count += 1
        return I_ext
