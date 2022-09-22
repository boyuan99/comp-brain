import numpy as np
from collections import OrderedDict
from compbrain.core import BaseComponent, CompBrainModelError


class IAFNeuron(BaseComponent):
    """Integrate and Fire Neuron Model"""

    def __init__(self, name, **kwargs):
        """
        initial function
        :param name: name for the integrate and fire neuron, every neuron in the circuit
            should be different
        :param kwargs: keyword arguments that overwrite initial conditions of state
            variables and values of parameters
        """
        super(IAFNeuron, self).__init__(name, **kwargs)
        self.params: OrderedDict = OrderedDict(
            V_T=-50.0, V_0=-80.0, V_imp=-20,
            C=1.0, dt=1e-5,
        )
        self.states: OrderedDict = OrderedDict(
            V=[-80],
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
        reset the integrate and fire neuron to its initial values
        """
        V_init = self.states['V'][0]
        self.states = OrderedDict(
            V=[V_init]
        )

    def compute(self, I_syn: float, I_ext: float, dt:float) -> dict:
        """
        Integrate and Fire gradient function

        :param I_syn: the input synapse current
        :param I_ext: the external injection current
        :param dt: time step
        :return: dict(V, N)
        """
        V = self.states['V'][-1]

        dt = dt * 1000
        V_T = self.params['V_T']
        V_0 = self.params['V_0']
        V_imp = self.params['V_imp']
        C = self.params['C']

        dV = I_ext/C

        V1 = V + dV * dt
        if V1 >= V_T:
            if V1 >= V_imp:
                V1 = V_0
            else:
                V1 = V_imp

        self.states['V'].append(V1)
        return self.states


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    neuron = IAFNeuron('L1')
    dt = 1e-5
    t = np.arange(0, 1, dt)

    I_ext = np.zeros_like(t)
    I_ext[int(0.2 * len(t)):int(0.8 * len(t))] = 1

    V = np.zeros_like(t)
    V[0] = 0.0

    for i in range(len(t)):
        if i < len(t)-1:
            neuron.compute(I_ext=I_ext[i], I_syn=0, dt=dt)

    plt.plot(t, neuron.states['V'])
    plt.show()
