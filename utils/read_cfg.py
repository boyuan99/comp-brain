import yaml
import numpy as np
from neurons import MorrisLecarNeuron
from synapses import CustomSynapse, InjectCurrent


def read_cfg(cfg):
    with open(cfg, 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_file.close()

    if 'times' in list(config):
        dt = config['times']['dt']
        steps = config['times']['steps']
        t = np.arange(0, dt*steps, dt)
    else:
        dt = 1e-4
        steps = 1e4
        t = np.arange(0, dt * steps, dt)

    if 'neurons' in list(config):
        neurons = read_neurons(config['neurons'])
    else:
        raise ValueError("no neuron defined in the config file")

    if 'synapses' in list(config):
        synapses = read_synapses(config['synapses'], t)
    else:
        raise ValueError("no synapse defined in the config file")

    return [neurons, synapses, t]


def read_neurons(neurons_cfg):
    """
    read the neurons part from the config file, instantiate each neuron and store them in a list
    params:
        neurons_cfg: config['neurons']

    return:
        list of instantiated neurons
    """
    neurons = []
    models = list(neurons_cfg)
    for model in models:
        if model == 'MorrisLecar':
            for neuron in list(neurons_cfg['MorrisLecar']):
                neurons.append(MorrisLecarNeuron(neuron, params=neurons_cfg['MorrisLecar'][neuron]))

        else:
            raise ValueError("no {} neurons implemented".format(model))

    return neurons


def read_synapses(synapses_cfg, t):
    synapses = []
    models = list(synapses_cfg)
    for model in models:
        if model == 'CustomSynapse':
            for synapse in list(synapses_cfg['CustomSynapse']):
                synapses.append(CustomSynapse(synapse, synapses_cfg['CustomSynapse'][synapse]['presynaptic'],
                                              synapses_cfg['CustomSynapse'][synapse]['postsynaptic'],
                                              params=synapses_cfg['CustomSynapse'][synapse]))

        elif model == 'InjectCurrent':
            for synapse in list(synapses_cfg['InjectCurrent']):
                synapses.append(InjectCurrent(synapse, t, synapses_cfg['InjectCurrent'][synapse]['presynaptic'],
                                              synapses_cfg['InjectCurrent'][synapse]['postsynaptic'],
                                              type=synapses_cfg['InjectCurrent'][synapse]['type']))

        else:
            raise ValueError("no {} synapses implemented".format(model))

    return synapses
