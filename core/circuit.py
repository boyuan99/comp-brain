import numpy as np
import yaml
from neurons import MorrisLecarNeuron
from synapses import BaseSynapse


class Circuit:
    def __init__(self, **kargs):
        if 'config_file' in kargs.keys():
            with open(kargs['config_file'], 'r') as config_file:
                self.config = yaml.safe_load(config_file)
                config_file.close()

        self.synapses = [synapse for synapse in list(self.config['synapses'])]
        self.neurons = [MorrisLecarNeuron(neuron) for neuron in list(self.config['neurons'])]
        self.name_scope = None

    def add_component(self, component):
        self.components.append(component)

    def reset_value(self):
        for component in self.components:
            component.reset_value(False)

    def component_count(self):
        return len(self.components)


default_circuit = Circuit()
