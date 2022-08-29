from core.node import BaseComponent
import numpy as np
from collections import OrderedDict


class CustomSynapse(BaseComponent):

    def __init__(self, name, presynaptic, postsynaptic, **kwargs):
        super(CustomSynapse, self).__init__(name, **kwargs)

        self.params: OrderedDict = OrderedDict(g_sat=0.15, k=0.05, n=1, t_delay=1, V_th=-50.5, V_rev=-50)
        self.states: OrderedDict = OrderedDict(I_syn=[])
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic

    def get_V_pre(self):
        V_pre = 0
        for i in range(len(self.parents)):
            V_pre += self.parents[i].states['V'][-1]

        return V_pre

    def get_V_post(self):
        V_post = 0
        for i in range(len(self.children)):
            V_post += self.children[i].states['V'][-1]

        return V_post

    def compute(self, V_pre, V_post):
        g_sat = self.params['g_sat']
        k = self.params['k']
        n = self.params['n']
        t_delay = self.params['t_delay']
        V_th = self.params['V_th']
        V_rev = self.params['V_rev']
        g = np.minimum(g_sat, k * np.maximum((V_pre * t_delay - V_th) ** n, 0))
        I_syn = g * (V_post - V_rev)

        self.states['I_syn'].append(I_syn)

        return I_syn

