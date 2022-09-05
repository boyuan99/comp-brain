import numpy as np
from collections import OrderedDict
from core import BaseComponent, CompBrainModelError


class PhotoInsensitiveNeuron(BaseComponent):
    """
    The Photo-Insensitive Cell Membrane Model
    'http://neurokernel.github.io/rfc/nk-rfc3.pdf'
    """

    def __init__(self, name, **kwargs):
        """
        initial function
        :param name: name for the photo insensitive neuron, every neuron in the circuit
            should be different
        :param kwargs: keyword arguments that overwrite initial conditions of state
            variables and values of parameters
        """
        super(PhotoInsensitiveNeuron, self).__init__(name, **kwargs)
        self.params: OrderedDict = OrderedDict(
            C=4.0, dt=1e-4,
            E_Cl=0.0, E_K=-70.0,
            g_L=0.006, g_K=0.082, g_A=1.6, g_dr=3.5, g_nov=3.0
        )
        self.states: OrderedDict = OrderedDict(
            V=[-67.5], Y2=[0.0], Y3=[0.0], Y4=[0.0], Y5=[0.0], Y6=[0.0]
        )

        """
        'init_V': -81.9925, 'init_sa': 0.2184, 'init_si': 0.9653,
        'init_dra': 0.0117, 'init_dri': 0.9998, 'init_nov': 0.0017,
        """

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
        Photo Insensitive gradient function

        :param I_syn: the input synapse current
        :param I_ext: the external injection current
        :param dt: time step
        :return: dict(V, Y2, Y3, Y4, Y5, Y6)
        """
        V = self.states['V'][-1]
        Y2 = self.states['Y2'][-1]
        Y3 = self.states['Y3'][-1]
        Y4 = self.states['Y4'][-1]
        Y5 = self.states['Y5'][-1]
        Y6 = self.states['Y6'][-1]

        dt = dt*1e3
        E_Cl = self.params['E_Cl']
        E_K = self.params['E_K']

        g_L = self.params['g_L']
        g_A= self.params['g_A']
        g_K = self.params['g_K']
        g_dr = self.params['g_dr']
        g_nov = self.params['g_nov']
        C = self.params['C']

        dV = (I_ext-g_K*(V-E_K)-g_L*(V-E_Cl)-g_A*Y2**3*Y3*(V-E_K)-
              g_dr*Y4**2*Y3*(V-E_K)-g_nov*Y6*(V-E_K))/C

        tau2 = 0.13 + 3.39*np.exp(-((-73-V)/20)**2)
        tau3 = 113*np.exp(-((-71-V)/29)**2)
        tau4 = 0.5 + (5.75*np.exp(-((-25-V)/32)**2))
        tau5 = 890
        tau6 = 3 + 106*np.exp(-((-20-V)/22)**2)

        dY2 = ((1/(1+np.exp((-23.7-V)/12.8)))**(1/3)-Y2)/tau2
        dY3 = (((0.9/(1+np.exp((-55-V)/-3.9)))+(0.1/(1+np.exp((-74.8-V)/-10.7))))-Y3)/tau3
        dY4 = ((1/(1+np.exp((-1-V)/9.1)))**(1/2)-Y4)/tau4
        dY5 = (1/(1+np.exp((-25.7-V)/-6.4))-Y5)/tau5
        dY6 = (1/(1+np.exp((-12-V)/11))-Y6)/tau6

        V1 = V + dV * dt
        Y21 = Y2 + dY2 * dt
        Y31 = Y3 + dY3 * dt
        Y41 = Y4 + dY4 * dt
        Y51 = Y5 + dY5 * dt
        Y61 = Y6 + dY6 * dt

        self.states['V'].append(V1)
        self.states['Y2'].append(Y21)
        self.states['Y3'].append(Y31)
        self.states['Y4'].append(Y41)
        self.states['Y5'].append(Y51)
        self.states['Y6'].append(Y61)

        return self.states
