import compbrain as cb

neurons, synapses, t = cb.utils.read_cfg('neurons_config.yaml')

circuit = cb.core.Circuit(neurons, synapses)
circuit.execute_circuit(t)

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 15))
for i in range(len(neurons)):
    plt.subplot(3, 4, i+1)
    plt.plot(t, neurons[i].states['V'], linewidth=2)
plt.show()

plt.figure(figsize=(20, 15))
for i in range(len(synapses)):
    plt.subplot(4, 6, i+1)
    if len(synapses[i].states['I_syn'])>0:
        plt.plot(t, synapses[i].states['I_syn'], linewidth=1)
plt.show()