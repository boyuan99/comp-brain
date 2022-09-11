import compbrain as cb
from compbrain import neurons, synapses, core
import numpy as np

n1 = neurons.MorrisLecarNeuron('n1')
s1 = synapses.InjectCurrent('s1', "None", 'n1')

c1 = core.Circuit([n1], [s1])
t = np.arange(0, 1, 1e-4)
c1.execute_circuit(t)


