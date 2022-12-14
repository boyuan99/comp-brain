import numpy as np
from collections import OrderedDict
from compbrain.core import BaseComponent, CompBrainModelError


class MorrisLecarNeuron(BaseComponent):
    """Morris Lecar Neuron Model"""

    def __init__(self, name, **kwargs):
        """
        initial function
        :param name: name for the morris lecar neuron, every neuron in the circuit
            should be different
        :param kwargs: keyword arguments that overwrite initial conditions of state
            variables and values of parameters
        """
        super(MorrisLecarNeuron, self).__init__(name, **kwargs)
        self.params: OrderedDict = OrderedDict(
            V_1=-15.0, V_2=2.0, V_3=-45.0, V_4=0.8,
            phi=0.0005, C=1, dt=1e-4, offset=50,
            E_L=-50.0, E_Ca=120.0, E_K=-75.0,
            g_L=0.1, g_Ca=2.0, g_K=7.0,
        )
        self.states: OrderedDict = OrderedDict(
            V=[-44.5], N=[0.5]
        )

        if 'params' in kwargs.keys():
            if not kwargs['params'] is None:
                for key, val in kwargs['params'].items():
                    if key in self.params:
                        self.params[key] = val
                    elif key in self.states:
                        self.states[key] = [val]
                    else:
                        raise CompBrainModelError(f"Unrecognized argument {key}")

    def get_I_syn(self) -> float:
        """
        get synapse current
        :return: synapse current
        """
        I_syn = 0
        for i in range(len(self.parents)):
            if len(self.parents[i].states['I_syn']) > 0:
                I_syn += self.parents[i].states['I_syn'][-1]

        return I_syn

    def get_I_ext(self) -> float:
        """
        get enternal injected current
        :return: injected current
        """
        I_ext = 0
        for i in range(len(self.parents)):
            if len(self.parents[i].states['I_ext']) > 0:
                I_ext += self.parents[i].states['I_ext'][-1]

        return I_ext

    def reset_value(self):
        """
        reset the morris lecar neuron to its initial values
        """
        V_init = self.states['V'][0]
        N_init = self.states['N'][0]
        self.states = OrderedDict(
            V=[V_init], N=[N_init]
        )

    def compute(self, I_syn: float, I_ext: float, dt:float) -> dict:
        """
        Morris-Lecar gradient function

        :param I_syn: the input synapse current
        :param I_ext: the external injection current
        :param dt: time step
        :return: dict(V, N)
        """
        V = self.states['V'][-1]
        N = self.states['N'][-1]
        if N < 1e-7:
            N = 0
        elif N > 1:
            N = 1

        dt = dt*1e3
        V_1 = self.params['V_1']
        V_2 = self.params['V_2']
        V_3 = self.params['V_3']
        V_4 = self.params['V_4']
        phi = self.params['phi']
        offset = self.params['offset']
        E_L = self.params['E_L']
        E_Ca = self.params['E_Ca']
        E_K = self.params['E_K']

        g_L = self.params['g_L']
        g_Ca = self.params['g_Ca']
        g_K = self.params['g_K']
        C = self.params['C']

        dV = (I_ext + offset - I_syn - g_L * (V - E_L) - 0.5 * g_Ca * (1 + np.tanh((V - V_1) / V_2)) * (V - E_Ca) - g_K * N * (
                    V - E_K)) / C
        dN = (0.5 * (1 + np.tanh((V - V_3) / V_4)) - N) * (phi * np.cosh((V - V_3) / (2 * V_4)))

        V1 = V + dV * dt
        N1 = N + dN * dt

        self.states['V'].append(V1)
        self.states['N'].append(N1)
        return self.states
