#!/usr/bin/env python

import os
import sys
sys.path.append('./_utils/')
import neat
from ExpEvoPyGame import exp_rewards
import pickle
import matplotlib.pyplot as plt


Behav_traject = True
Acc_motors = False

## LOAD WINNER GENOME AND SHOW DEMO ########################################################

print 'This is how the best robot does:'

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-nn')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

with open('winner', 'rb') as f:
    gen = pickle.load(f)

print gen

winner_net = neat.ctrnn.CTRNN.create(gen, config, .01)

num_trials = 3
for j in range(2*num_trials):
    type =""
    if j < num_trials:
        type = "predator"
    elif j >= num_trials:
        type = "prey"

    winner_net.reset()
    exp_winner = exp_rewards(net=winner_net, j=j, n_rew=0, type=type, f_size=5.)
    f = exp_winner.run()


    if Acc_motors:
        ml = f[2][:,0]
        mr = f[2][:,1]
        n413 = f[2][:,2]

        #plt.figure(figsize=(30,7))
        plt.plot(range(len(ml[1:])), ml[1:], label='ML')
        plt.plot(range(len(ml[1:])), mr[1:], label='MR')
        #plt.plot(range(len(ml[1:])), n413[1:], label='n413')
        plt.ylim((0, 1.1))
        plt.title('Output of the motoneurons through time')
        plt.xlabel('Time steps')
        plt.ylabel('Output y')
        plt.legend()
        plt.show()


    if Behav_traject:
        pos_evo = f[1][0]
        pos_opp = f[1][1]

        plt.plot(pos_evo[:, 0], pos_evo[:, 1], label='Evo_Agent')
        plt.plot(pos_opp[:, 0], pos_opp[:, 1], label=type)
        plt.ylim((0, 10))
        plt.xlim((0, 10))
        plt.title('Behavioral trajectories for both agents')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.legend()
        plt.show()



