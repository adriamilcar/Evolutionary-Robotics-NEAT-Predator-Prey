#!/usr/bin/env python

import sys
sys.path.append('./_utils/')
import pickle
from R import *
from MI_body import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib



accumulated = True

## LOAD WINNER GENOME AND COMPUTE R ########################################################

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-nn')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

with open('winner', 'rb') as f:
    gen = pickle.load(f)

print gen

r = compute_R(gen, 300)
print r


if accumulated:
    r_acc = []
    r_acc_ind = []
    i_body = []
    for max_iter in range(1, 301):  #range(1, 301)
        r_temp = compute_R(gen, max_iter)

        r_acc.append(r_temp[0])

        if max_iter == 1:
            r_acc_ind = r_temp[1].values()
        else:
            r_acc_ind = np.vstack((r_acc_ind, r_temp[1].values()))

        i_body.append(compute_I(gen, max_iter)[0])



    plt.plot(range(300), r_acc, label='R', color='g')  #range(300)
    plt.plot(range(300), i_body, label='I(Body;Identity)', color='b')  # range(300)
    plt.title("Within-trial accumulation of R and I(Body;Identity)")
    plt.xlabel("Time steps")
    plt.ylabel("Bits")
    plt.legend()
    plt.savefig('RandBody_Acc.png', bbox_inches='tight')
    plt.clf()
    #plt.show()

    labels = ['ML', 'MR', 'n413', 'n406']
    for ii, neuron in enumerate(labels):
        plt.plot(range(300), r_acc_ind[:, ii], label=neuron) #range(300)

    plt.title("Within-trial accumulation of R for each neuron")
    plt.xlabel("Time steps")
    plt.ylabel("Bits")
    plt.legend()
    plt.savefig('RNeurons_Acc.png', bbox_inches='tight')
    plt.clf()



    r_diff = np.diff(r_acc)
    i_body_diff = np.diff(i_body)
    plt.plot(range(299), r_diff, label='R', color='r') #range(299)
    plt.plot(range(299), i_body_diff, label='I(Body;Identity)', color='black')  # range(299)
    plt.title("Within-trial changes in R and I(Body;Identity)")
    plt.xlabel("Time steps")
    plt.ylabel("Bits")
    plt.legend()
    plt.savefig('RandBody_Changes.png', bbox_inches='tight')
    plt.clf()
    #plt.show()




matplotlib.rcParams.update({'font.size': 11})
width = 0.1       # the width of the bars

fig, ax = plt.subplots()

totalR = ax.bar((0), r[0], width, color='green')

ML_id = ax.bar((width*1.7), r[1][0], width, color='#A80000')  #or color #280ABF

MR_id = ax.bar((width*3.4), r[1][1], width, color='#A80000')

n413_id = ax.bar((width*5.1), r[1][413], width, color='#A80000')

n406_id = ax.bar((width*6.8), r[1][406], width, color='#A80000')


# add some text for labels, title and axes ticks
ax.set_ylabel("R in bits")
ax.set_title("Other's Identity represented by the neural network")
plt.xticks([width*0, width*1.7, width*3.4, width*5.1, width*6.8],['Network', 'ML', 'MR', 'n413', 'n406']) #['Network']+r[1].keys())

plt.xlim(-width*1., width*7.8)
plt.ylim(0, 0.3)
plt.show()

fig.savefig("R_2D.png", bbox_inches="tight")

