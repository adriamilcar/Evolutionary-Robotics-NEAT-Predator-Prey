import os
import sys
sys.path.append('./_utils/')
sys.path.append('./neat')
import __init__ as neat
sys.modules["neat"] = neat
import Box2DWorld
from RobotSetupEvo import Setup
import numpy as np
from numpy import *



def bin_int_conv(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


def R(distr):
    pS = np.sum(distr, axis=(1, 2))
    pB = np.sum(distr, axis=(0, 2))
    pE = np.sum(distr, axis=(0, 1))
    pSBE = distr
    pSE = np.sum(distr, axis=1)
    pSB = np.sum(distr, axis=2)

    np.seterr(divide='ignore', invalid='ignore')

    env_entropy = -np.sum(pE * ma.log2(pE).filled(0))
    brain_entropy = -np.sum(pB * ma.log2(pB).filled(0))
    sensors_entropy = -np.sum(pS * ma.log2(pS).filled(0))
    joint_entropy = -np.sum(pSBE * ma.log2(pSBE).filled(0))
    mutual_SensEnv = np.sum(pSE * ma.log2(pSE / (pE[np.newaxis, :] * pS[:, np.newaxis])).filled(0))
    mutual_SensBrain = np.sum(pSB * ma.log2(pSB / (pB[np.newaxis, :] * pS[:, np.newaxis])).filled(0))

    multi_info = env_entropy + brain_entropy + sensors_entropy - joint_entropy
    r = multi_info - mutual_SensEnv - mutual_SensBrain
    return r


def compute_R(winner, max_iter=300, random=False):

    print "*------------------*****************-------------------------*"
    print "*------------------** COMPUTING R **-------------------------*"
    print "*------------------*****************-------------------------*"

    local_dir = os.path.dirname(__file__)
    conf = 'config-nn'
    if random:
        conf = 'config-random'
    config_path = os.path.join(local_dir, conf)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

    net = neat.ctrnn.CTRNN.create(winner, config, .01)

    distr = np.zeros((2**len(net.input_nodes), 2**winner.size()[0], 2))
    # 3D array storing joint probability distribution p(Sens,Brain,Env)

    brain_keys = [k for (k,v) in net.values[1-net.active].items() if k >= 0]
    indiv_distr = {key: np.zeros((2**len(net.input_nodes), 2, 2)) for key in brain_keys}
    # for each node: [R(E_predator/prey : Bi | S),  R(E_seen/no-seen : Bi | S)]

    num_trials = 3   # number of predator+prey trials to compute R
    total_timesteps = 0.
    fit = 0.

    for ii in range(2*num_trials):
        type = ""
        e = None
        if ii < num_trials:
            type = "predator"
            e = 0
        elif ii >= num_trials:
            type = "prey"
            e = 1

        net.reset()
        exp = Setup(net=net, j=ii, type=type, field_size=5.)
        i = 0
        while i < max_iter and not exp.stop:
            flag = Box2DWorld.destroy([exp.objs, exp.epucks])
            exp.update(i)
            Box2DWorld.step()

            sens = np.array([v for (k,v) in net.values[1-net.active].items() if k < 0])
            sens_state = np.piecewise(sens, [sens <= .5, sens > .5], [0, 1])  # binarization

            nodes = np.array([v for (k,v) in net.values[1-net.active].items() if k >= 0])
            brain_state = np.piecewise(nodes, [nodes <= .5, nodes > .5], [0, 1])

            s = bin_int_conv(sens_state.astype(int).tolist())
            b = bin_int_conv(brain_state.astype(int).tolist())

            distr[s, b, e] += 1.

            for k, value in enumerate(brain_state.astype(int)):
                indiv_distr[brain_keys[k]][s, value, e] += 1.  # predator or prey

            i += 1
            total_timesteps += 1.

        ff = exp.epucks[0].time
        if type == "predator":  # EvoAgent has to avoid predators
            fit += ff
        elif type == "prey":  # EvoAgent has to catch prey
            fit += (300. - ff)

        exp.reset()
        flag = Box2DWorld.destroy([exp.objs, exp.epucks])
        print type
        print i

    distr /= total_timesteps
    for key in brain_keys:
        indiv_distr[key] /= total_timesteps


    total_r = np.around(R(distr), decimals=4)
    distributed_r = {bk: np.around(R(indiv_distr[bk]), decimals=4) for bk in brain_keys}

    return [total_r, distributed_r, fit/(2.*num_trials)]
