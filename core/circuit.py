import numpy as np
import yaml


class Circuit:
    def __init__(self, neurons, synapses, **kargs):
        self.neurons = neurons
        self.synapses = synapses

        for synapse in synapses:
            pre_synaptic_neurons = self.find_neuron(synapse.presynaptic)
            post_synaptic_neurons = self.find_neuron(synapse.postsynaptic)

            pre_synaptic_neurons.append_children(synapse)
            post_synaptic_neurons.append_parents(synapse)
            synapse.append_children(post_synaptic_neurons)
            synapse.append_parents(pre_synaptic_neurons)

    def find_neuron(self, name):
        for i in range(len(self.neurons)):
            if self.neurons[i].name == name:
                return self.neurons[i]

        raise ValueError("Couldn't find neuron")
        return None

    def execute_step(self, dt=1e-4, synapses_policy=True, neurons_policy=True):
        if synapses_policy:
            for synapse in self.synapses:
                V_pre = synapse.get_V_pre()
                V_post = synapse.get_V_post()
                _ = synapse.compute(V_pre, V_post)

        if neurons_policy:
            for neuron in self.neurons:
                I_syn = neuron.get_I_syn()
                _, _ = neuron.compute(I_syn, dt)

    def execute_circuit(self, t):
        dt = t[1] - t[0]

        for i in range(len(t)):
            if i < len(t)-1:
                self.execute_step(dt)
            else:
                self.execute_step(dt, synapses_policy=True, neurons_policy=False)
