import yaml
import numpy as np
from compbrain.neurons import MorrisLecarNeuron, PhotoInsensitiveNeuron, HodgkinHuxleyNeuron
from compbrain.synapses import CustomSynapse, InjectCurrent
from compbrain.core import CompBrainUtilsError


def read_cfg(cfg):
    """

    :param cfg: the configuration file
    :return: [neurons, synapses, t]
    """
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
        raise CompBrainUtilsError("no neuron defined in the config file")

    if 'synapses' in list(config):
        synapses = read_synapses(config['synapses'], t)
    else:
        raise CompBrainUtilsError("no synapse defined in the config file")

    return [neurons, synapses, t]


def read_neurons(neurons_cfg):
    """
    read the neurons part from the config file, instantiate each neuron and store them in a list
    :param:
        neurons_cfg: config['neurons']

    :return:
        list of instantiated neurons
    """
    neurons = []
    models = list(neurons_cfg)
    for model in models:
        if model == 'MorrisLecar':
            for neuron in list(neurons_cfg['MorrisLecar']):
                neurons.append(MorrisLecarNeuron(neuron, params=neurons_cfg['MorrisLecar'][neuron]))

        elif model == 'PhotoInsensitive':
            for neuron in list(neurons_cfg['PhotoInsensitive']):
                neurons.append(PhotoInsensitiveNeuron(neuron, params=neurons_cfg['PhotoInsensitive'][neuron]))

        elif model == 'HodgkinHuxley':
            for neuron in list(neurons_cfg['HodgkinHuxley']):
                neurons.append(HodgkinHuxleyNeuron(neuron, params=neurons_cfg['HodgkinHuxley'][neuron]))

        else:
            raise CompBrainUtilsError("no {} neurons implemented".format(model))

    return neurons


def read_synapses(synapses_cfg, t: np.ndarray):
    """
    read the synapses part from the configuration file, instantiate each
    synapse and store them in a list

    :param synapses_cfg: the synapse config part
    :param t: a time numpy array for creating the external current
    :return: list of instantiated synapses
    """
    synapses = []
    models = list(synapses_cfg)
    for model in models:
        if model == 'CustomSynapse':
            for synapse in list(synapses_cfg['CustomSynapse']):
                synapses.append(CustomSynapse(synapse, synapses_cfg['CustomSynapse'][synapse]['presynaptic'],
                                              synapses_cfg['CustomSynapse'][synapse]['postsynaptic'],
                                              params=synapses_cfg['CustomSynapse'][synapse]['params']))

        elif model == 'InjectCurrent':
            for synapse in list(synapses_cfg['InjectCurrent']):
                synapses.append(InjectCurrent(synapse, t, synapses_cfg['InjectCurrent'][synapse]['presynaptic'],
                                              synapses_cfg['InjectCurrent'][synapse]['postsynaptic'],
                                              type=synapses_cfg['InjectCurrent'][synapse]['type']))

        else:
            raise ValueError("no {} synapses implemented".format(model))

    return synapses
