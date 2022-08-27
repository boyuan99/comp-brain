from core.node import BaseComponent
import numpy as np
from collections import OrderedDict


class MorrisLecarNeuron(BaseComponent):
    """Morris Lecar Neuron Model"""
    # Default State Variables of the Morris-Lecar Model
    Default_States = OrderedDict(
        V=-52.5, N=0.02
    )
    # Default Parameters of the Morris-Lecar Model
    Default_Params = OrderedDict(
        V_1=30.0, V_2=15.0, V_3=0.0, V_4=30.0,
        phi=0.025, C=6.698, dt=1e-4,
        E_L=-50.0, E_Ca=100.0, E_K=-70.0,
        g_L=0.5, g_Ca=1.1, g_K=2.0,
    )

    def __init__(self, **kwargs):
        super(MorrisLecarNeuron, self).__init__()
        self.params = OrderedDict(self.Default_Params.copy())
        self.value = [self.Default_States['V'], self.Default_States['N']]
        self.value_array = self.value_array.append(self.value)

        for key, val in kwargs.items():
            if key in self.params:
                self.params[key] = val
            elif key in self.states:
                self.states[key] = val
                self.initial_states[key] = val
            else:
                raise err.CompNeuroModelError(f"Unrecognized argument {key}")

    def compute(self, I_syn):
        V = self.value[0]
        N = self.value[1]

        dt = self.params['dt'] * 1e3
        V_1 = self.params['V_1']
        V_2 = self.params['V_2']
        V_3 = self.params['V_3']
        V_4 = self.params['V_4']
        phi = self.params['phi']
        E_L = self.params['E_L']
        E_Ca = self.params['E_Ca']
        E_K = self.params['E_K']

        g_L = self.params['g_L']
        g_Ca = self.params['g_Ca']
        g_K = self.params['g_K']
        C = self.params['C']

        dV = (I_syn - g_L * (V - E_L) - 0.5 * g_Ca * (1 + np.tanh((V - V_1) / V_2)) * (V - E_Ca) - g_K * N * (
                    V - E_K)) / C
        dN = (0.5 * (1 + np.tanh((V - V_3) / V_4)) - N) * (phi * np.cosh((V - V_3) / (2 * V_4)))

        V1 = V + dV * dt
        N1 = N + dN * dt

        self.value = [V1, N1]
        self.value_array = self.value_array.append(self.value)
        return [V1, N1]
