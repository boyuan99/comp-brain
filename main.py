import yaml
from core import Circuit
from neurons import MorrisLecarNeuron

with open("./neurons_config.yaml", 'r') as config_file:
    config = yaml.safe_load(config_file)
    config_file.close()

neurons = []
for neuron in list(config['neurons']):
    neurons.append(MorrisLecarNeuron(neuron, params=config['neurons'][neuron]))

