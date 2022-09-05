import numpy as np
from collections import OrderedDict
from core import BaseComponent, CompBrainModelError


class HodgkinHuxleyNeuron(BaseComponent):
    """Hodgkin Huxley Neuron Model"""

    def __init__(self, name, **kwargs):
        """
        initial function
        :param name: name for the morris lecar neuron, every neuron in the circuit
            should be different
        :param kwargs: keyword arguments that overwrite initial conditions of state
            variables and values of parameters
        """
        super(HodgkinHuxleyNeuron, self).__init__(name, **kwargs)
        self.params: OrderedDict = OrderedDict(
            g_Na=120.0, g_K=36.0, g_L=0.3,
            E_Na=50.0, E_K=-77.0, E_L=-54.387,
            offset=0, C=1,
        )
        self.states: OrderedDict = OrderedDict(
            V=[-60], n=[0.0], m=[0.0], h=[1.0]
        )

        if ('params' in kwargs.keys()) & (not kwargs['params'] is None):
            for key, val in kwargs['params'].items():
                if key in self.params:
                    self.params[key] = val
                elif key in self.states:
                    self.states[key] = val
                    self.initial_states[key] = val
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

    def compute(self, I_syn: float, I_ext: float, dt:float) -> dict:
        """
        Morris-Lecar gradient function

        :param I_syn: the input synapse current
        :param I_ext: the external injection current
        :param dt: time step
        :return: dict(V, m, n, h)
        """
        V = self.states['V'][-1]
        m = self.states['m'][-1]
        n = self.states['n'][-1]
        h = self.states['h'][-1]

        dt = dt*1e3
        offset = self.params['offset']
        E_L = self.params['E_L']
        E_Na = self.params['E_Na']
        E_K = self.params['E_K']
        g_L = self.params['g_L']
        g_Na = self.params['g_Na']
        g_K = self.params['g_K']
        C = self.params['C']

        dV = (offset + I_ext - I_syn - g_K*n**4*(V-E_K) - g_Na*m**3*h*(V-E_Na) -
              g_L*(V-E_L))/C

        dn = (0.01*(10-V)/(np.exp(1-V/10)-1))*(1-n) - 0.125*np.exp(-V/80)*n
        dm = ((2.5-0.1*V)/(np.exp(2.5-V/10)-1))*(1-m) - 4*np.exp(-V/18)*m
        dh = 0.07*np.exp(-V/20)*(1-h) - h/(1+np.exp(3-V/10))

        V1 = V + dV * dt
        n1 = n + dn * dt
        m1 = m + dm * dt
        h1 = h + dh * dt

        self.states['V'].append(V1)
        self.states['n'].append(n1)
        self.states['m'].append(m1)
        self.states['h'].append(h1)
        return self.states
