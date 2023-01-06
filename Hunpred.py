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


# H(distr[axis=1] | distr[axis=0])
def h_unpred(distr):
    pS = np.sum(distr, axis=1)
    pM = np.sum(distr, axis=0)
    pSM = distr

    np.seterr(divide='ignore', invalid='ignore')

    M_entropy = -np.sum(pM * ma.log2(pM).filled(0))
    h = np.sum(pSM * ma.log2(pS[:,np.newaxis] / pSM).filled(0))
    return h, M_entropy


# computes H(Motors_t+1 | Sens_t)
def compute_Hunpred(winner, max_iter=300, random=False):

    print "*------------------*****************-------------------------*"
    print "*------------------** COMPUTING H **-------------------------*"
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

    distr = np.zeros((2**len(net.input_nodes), 2**2))
    # 2D array storing joint probability distribution p(Sens_t, Motors_t+1)


    num_trials = 3   # number of predator+prey trials to compute R
    total_timesteps = 0.
    fit = 0.
    sens_state = None

    for ii in range(2*num_trials):
        type = ""
        if ii < num_trials:
            type = "predator"
        elif ii >= num_trials:
            type = "prey"

        net.reset()
        exp = Setup(net=net, j=ii, type=type, field_size=5.)
        i = 0
        while i < max_iter and not exp.stop:
            flag = Box2DWorld.destroy([exp.objs, exp.epucks])
            exp.update(i)
            Box2DWorld.step()


            if i > 0:
                nodes = np.array([v for (k,v) in net.values[1-net.active].items() if k==0 or k==1])
                motor_state = np.piecewise(nodes, [nodes <= .5, nodes > .5], [0, 1])

                s = bin_int_conv(sens_state.astype(int).tolist())
                m = bin_int_conv(motor_state.astype(int).tolist())

                distr[s, m] += 1.

            sens = np.array([v for (k, v) in net.values[1 - net.active].items() if k < 0])
            sens_state = np.piecewise(sens, [sens <= .5, sens > .5], [0, 1])

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

    h = h_unpred(distr)
    total_hunpred = np.around(h[0], decimals=4)
    motor_entropy = np.around(h[1], decimals=4)

    return [total_hunpred, motor_entropy, fit/(2.*num_trials)]
