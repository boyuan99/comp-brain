import yaml
import numpy as np
from core import Circuit
from neurons import MorrisLecarNeuron
from synapses import CustomSynapse
from utils import read_cfg

neurons, synapses, t = read_cfg('neurons_config_noL.yaml')

circuit = Circuit(neurons, synapses)
circuit.execute_circuit(t)

import matplotlib.pyplot as plt
plt.plot(t, neurons[0].states['V'])
plt.show()