from core import Circuit
from utils import read_cfg

neurons, synapses, t = read_cfg('neurons_config.yaml')

circuit = Circuit(neurons, synapses)
circuit.execute_circuit(t)

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 15))
for i in range(len(neurons)):
    plt.subplot(2, 4, i+1)
    plt.plot(t, neurons[i].states['V'], linewidth=5)
plt.show()

plt.figure(figsize=(20, 15))
for i in range(len(synapses)):
    plt.subplot(3, 6, i+1)
    if len(synapses[i].states['I_syn'])>0:
        plt.plot(t, synapses[i].states['I_syn'])
plt.show()