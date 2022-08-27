import yaml
import numpy as np

with open("./neurons_config.yaml", 'r') as config_file:
    config = yaml.safe_load(config_file)
    config_file.close()

# neurons = []
# for neuron in list(config['neurons']):
#     neurons.append(MorrisLecarNeuron(name=neuron))

synapses = []
for synapse in list(config['synapses']):
    synapses.append()

