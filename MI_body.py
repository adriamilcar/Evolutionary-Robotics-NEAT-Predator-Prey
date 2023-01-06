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
import itertools



def bin_int_conv(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


def mutual_info(distr):
    pB = np.sum(distr, axis=1)
    pE = np.sum(distr, axis=0)
    pBE = distr

    np.seterr(divide='ignore', invalid='ignore')

    mutual_BodyEnv = np.sum(pBE * ma.log2(pBE / (pE[np.newaxis, :] * pB[:, np.newaxis])).filled(0))

    return mutual_BodyEnv


def compute_I(winner, max_iter):

    print "*------------------*****************-------------------------*"
    print "*------------------** COMPUTING I **-------------------------*"
    print "*------------------*****************-------------------------*"

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-nn')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

    net = neat.ctrnn.CTRNN.create(winner, config, .01)

    distr = np.zeros((2**4, 2))
    # 3D array storing joint probability distribution p(BodyPos,Env)

    indiv_distr = {key: np.zeros((2, 2)) for key in range(16)}
    # for each node: [I(E_predator/prey : BodyPos),  I(E_startPos : BodyPos)]

    num_trials = 3   # number of predator+prey trials to compute I
    f_size = 5.
    total_timesteps = 0.
    fit = 0.

    for ii in range(2*num_trials):
        type =""
        env_state = 0
        if ii < num_trials:
            type = "predator"
        elif ii >= num_trials:
            type = "prey"
            env_state = 1

        net.reset()
        exp = Setup(net=net, j=ii, type=type, field_size=f_size)
        i = 0
        while i < max_iter and not exp.stop:  # compute I(BodyPos : Environment) for the first XX timesteps (time before categorization)
            flag = Box2DWorld.destroy([exp.objs, exp.epucks])
            exp.update(i)
            Box2DWorld.step()

            body_state = []
            bodyPos = exp.epucks[0].getPosition()
            xpos = bodyPos[0]
            ypos = bodyPos[1]

            for pos in [ypos, xpos]:
                if pos < -(f_size / 2.):
                    body_state.append([0, 0])
                elif pos < 0:
                    body_state.append([0, 1])
                elif pos < (f_size / 2.):
                    body_state.append([1, 0])
                else:
                    body_state.append([1, 1])

            b = bin_int_conv(list(itertools.chain(*body_state)))
            e = env_state

            distr[b, e] += 1.

            for jj in range(16):
                v=0
                if jj==b:
                    v=1
                indiv_distr[jj][v, env_state] += 1.  # predator or prey

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
    for key in range(16):
        indiv_distr[key] /= total_timesteps


    total_info = np.around(mutual_info(distr), decimals=4)
    distributed_info = {jj: np.around(mutual_info(indiv_distr[jj]), decimals=4) for jj in range(16)}

    return [total_info, distributed_info, fit/(2.*num_trials)]
