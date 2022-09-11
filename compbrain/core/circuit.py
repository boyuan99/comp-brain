import numpy as np
from tqdm import tqdm


class Circuit:
    """
    Neuron circuit base class

    execute step by step
    execute the whole circuit: update synapses first, then update neurons

    :argument
        neurons: list of instantiated neurons
        synapses: list of instantiated synapses

    """
    def __init__(self, neurons: list, synapses: list, **kargs):
        self.neurons = neurons
        self.synapses = synapses

        for synapse in synapses:
            pre_synaptic_neurons = self.find_neuron(synapse.presynaptic)
            post_synaptic_neurons = self.find_neuron(synapse.postsynaptic)

            if pre_synaptic_neurons:
                pre_synaptic_neurons.append_children(synapse)
                synapse.append_parents(pre_synaptic_neurons)

            if post_synaptic_neurons:
                post_synaptic_neurons.append_parents(synapse)
                synapse.append_children(post_synaptic_neurons)

    def find_neuron(self, name: str):
        """
        Find the instantiated neuron according its name
        :param name: the str type name for the neuron.
        :return: instantiated neuron
        """
        if name != "None":
            for i in range(len(self.neurons)):
                if self.neurons[i].name == name:
                    return self.neurons[i]

            raise ValueError("Couldn't find neuron")
        return None

    def execute_step(self, dt: float=1e-4, synapses_policy: bool=True, neurons_policy: bool=True):
        """
        execute the whole circuit by one time step
        :param dt: dt
        :param synapses_policy: whether execute the synapses
        :param neurons_policy: whether execute the neurons
        :return: None
        """
        if synapses_policy:
            for synapse in self.synapses:
                V_pre = synapse.get_V_pre()
                V_post = synapse.get_V_post()
                _ = synapse.compute(V_pre, V_post)

        if neurons_policy:
            for neuron in self.neurons:
                I_syn = neuron.get_I_syn()
                I_ext = neuron.get_I_ext()
                _ = neuron.compute(I_syn, I_ext, dt)

    def execute_circuit(self, t: np.ndarray):
        """
        execute the whole circuit for all the time steps
        :param t: a time numpy array
        :return: None
        """
        dt = t[1] - t[0]

        for i in tqdm(range(len(t))):
            if i < len(t)-1:
                self.execute_step(dt)
            else:
                self.execute_step(dt, synapses_policy=True, neurons_policy=False)
