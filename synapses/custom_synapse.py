from core.node import BaseComponent
import numpy as np
from collections import OrderedDict


class CustomSynapse(BaseComponent):
    """
    A implementation of customized synapse model from "http://neurokernel.github.io/rfc/nk-rfc2.pdf"

    :argument
        name: the name of the synapse, notice that in a circuit all of the
            synapse's names should be different
        presynaptic: the name of the presynaptic neuron, doesn't need to be instantiated
        postsynaptic: the name of the postsynaptic neuron, doesn't need to be instantiated
        kwargs: keyword arguments that overwrite initial conditions of state
            variables and values of parameters
    """

    def __init__(self, name: str, presynaptic: str, postsynaptic: str, **kwargs):
        super(CustomSynapse, self).__init__(name, **kwargs)

        self.params: OrderedDict = OrderedDict(g_sat=0.15, k=0.05, n=1, t_delay=1, V_th=-50.5, V_rev=-50)
        self.states: OrderedDict = OrderedDict(I_ext=[], I_syn=[])
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic

    def get_V_pre(self) -> float:
        """
        get the presynaptic voltage
        :return:
        """
        V_pre = 0
        for i in range(len(self.parents)):
            V_pre += self.parents[i].states['V'][-1]

        return V_pre

    def get_V_post(self) -> float:
        V_post = 0
        for i in range(len(self.children)):
            V_post += self.children[i].states['V'][-1]

        return V_post

    def compute(self, V_pre: float, V_post: float) -> dict:
        """
        custom synapse function

        :param V_pre: presynaptic neuron voltage
        :param V_post: post synaptic neuron voltage
        :return: dict(V, N)
        """
        g_sat = self.params['g_sat']
        k = self.params['k']
        n = self.params['n']
        t_delay = self.params['t_delay']
        V_th = self.params['V_th']
        V_rev = self.params['V_rev']
        g = np.minimum(g_sat, k * np.maximum((V_pre * t_delay - V_th) ** n, 0))
        I_syn = g * (V_post - V_rev)

        self.states['I_syn'].append(I_syn)

        return self.states

