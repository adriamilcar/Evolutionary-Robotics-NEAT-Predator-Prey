#!/usr/bin/env python

import sys
sys.path.append('./_utils/')
import pickle
from Hunpred import *
import numpy as np
import matplotlib.pyplot as plt


accumulated = True

## LOAD WINNER GENOME AND COMPUTE H ########################################################

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-nn')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

with open('winner', 'rb') as f:
    gen = pickle.load(f)

print gen

h = compute_Hunpred(gen, 300)
print h


if accumulated:
    h_acc = np.array([compute_Hunpred(gen, max_iter)[0:2] for max_iter in range(1, 301)])


    plt.plot(range(300), h_acc[:,0], label='H_unpred')  #range(300)
    plt.plot(range(300), h_acc[:,1], label='Motor entropy')  #range(300)
    plt.plot(range(300), h_acc[:,1]-h_acc[:,0], label='I_pred')  # range(300)
    plt.title("Within-trial accumulation of H_unpred")
    plt.xlabel("Time steps")
    plt.ylabel("Bits")
    plt.ylim((0, 2))
    plt.legend()
    plt.savefig('Hunpred_Acc.png', bbox_inches='tight')
    plt.clf()


    h_diff = np.diff(h_acc[:,0])
    plt.plot(range(299), h_diff, color='r') #range(299)
    plt.title("Within-trial changes in H_unpred")
    plt.xlabel("Time steps")
    plt.ylabel("Bits")
    plt.ylim((-0.15, 1))
    plt.savefig('Hunpred_Changes.png', bbox_inches='tight')
    plt.clf()
