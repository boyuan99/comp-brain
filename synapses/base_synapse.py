from core.node import BaseComponent
import numpy as np
from collections import OrderedDict


class BaseSynapse(BaseComponent):
    Default_Params: OrderedDict = OrderedDict()
    """A dictionary of (name, value) pair for parameters"""

    Default_States: OrderedDict = OrderedDict(g_sat=0.15, k=0.05, n=1, t_delay=1, V_th=-50.5, V_rev=-50)
    """A dictionary of state variables"""
    def __init__(self):
        super(BaseSynapse, self).__init__()

        self.V_pre = self.get_parents().value
        self.V_post = self.get_children().value

    def compute(self):
        g_sat = self.params['g_sat']
        k = self.params['k']
        n = self.params['n']
        t_delay = self.params['t_delay']
        V_th = self.params['V_th']
        V_rev = self.params['V_rev']
        g = np.minimum(g_sat, k * np.maximum((self.V_pre * t_delay - V_th) ** n, 0))
        I_syn = -g * (self.V_post - V_rev)

        self.value = I_syn
        self.value_array.append(I_syn)

        return I_syn

