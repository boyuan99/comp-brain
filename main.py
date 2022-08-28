import yaml
import numpy as np
from core import Circuit
from neurons import MorrisLecarNeuron
from synapses import BaseSynapse

with open("./neurons_config.yaml", 'r') as config_file:
    config = yaml.safe_load(config_file)
    config_file.close()

neurons = []
for neuron in list(config['neurons']):
    neurons.append(MorrisLecarNeuron(neuron, params=config['neurons'][neuron]))

synapses = []
for synapse in list(config['synapses']):
    synapses.append(BaseSynapse(synapse, config['synapses'][synapse]['presynaptic'],
                                config['synapses'][synapse]['postsynaptic'], params=config['synapses'][synapse]))

circuit = Circuit(neurons, synapses)
t = np.arange(0, 1, 1e-4)
circuit.execute_circuit(t)
