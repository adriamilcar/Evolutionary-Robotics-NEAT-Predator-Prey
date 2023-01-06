#!/usr/bin/env python

import os
import sys
sys.path.append('./_utils/')
import neat
from neat.six_util import iteritems
import Box2DWorld
from RobotSetupEvo import Setup
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


## LOAD WINNER GENOME AND COMPUTE R ########################################################

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-nn')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

with open('winner', 'rb') as f:
    gen = pickle.load(f)

print gen

net = neat.ctrnn.CTRNN.create(gen, config, .01)

brain_keys = [k for (k,v) in net.values[1-net.active].items() if k >= 0]

num_trials = 3   # number of predator+prey trials to compute R
total_timesteps = 0.

fig = plt.figure(figsize=(20,20))
ax = fig.gca(projection='3d')

for ii in range(2 * num_trials):
    color = ""
    type = ""
    if ii < num_trials:
        type = "predator"
        color = "red"
    elif ii >= num_trials:
        type = "prey"
        color = "blue"

    s = np.array([0., 0., 0.])  # 3D state being tracked

    net.reset()
    exp = Setup(net=net, j=ii, type=type, field_size=5.)
    i = 0
    while i < 500 and not exp.stop:
        flag = Box2DWorld.destroy([exp.objs, exp.epucks])
        exp.update(i)
        Box2DWorld.step()

        #sens = np.array([v for (k, v) in net.values[1 - net.active].items() if k < 0])
        #sens_state = np.piecewise(sens, [sens <= .5, sens > .5], [0, 1])  # binarization

        ss=[]
        for node_key, ne in iteritems(net.node_evals):
            node_inputs = [net.values[1-net.active][z] * w for z, w in ne.links]
            ss.append(ne.aggregation(node_inputs))

        nodes = np.array([v for (k, v) in net.values[1 - net.active].items() if k >= 0])
        brain_state = np.piecewise(nodes, [nodes <= .5, nodes > .5], [0, 1])

        s = np.vstack((s, ss[:3]))  #before: nodes[-3:]

        i += 1
        total_timesteps += 1.

    exp.reset()
    flag = Box2DWorld.destroy([exp.objs, exp.epucks])

    if ii==0 or ii==3:
        ax.plot(s[:, 0], s[:, 1], s[:, 2], color=color, linewidth=2, label=type)
    else:
        ax.plot(s[:, 0], s[:, 1], s[:, 2], color=color, linewidth=2)


    #plt.quiver(s[:-1, 0], s[:-1, 1], s[1:, 0] - s[:-1, 0], s[1:, 1] - s[:-1, 1], scale_units='xy', angles='xy', scale=1, color=color)

ax.set_xlabel('neuron ML')
ax.set_ylabel('neuron MR')
ax.set_zlabel('neuron ' + str(brain_keys[2]))
ax.set_title('Phase space trajectories')
plt.legend()

fig.savefig('DS_allTrials.png', bbox_inches='tight')
plt.show()

