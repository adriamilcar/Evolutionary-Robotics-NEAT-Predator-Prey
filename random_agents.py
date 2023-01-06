#!/usr/bin/env python

import sys
sys.path.append('./_utils/')
import neat
from R import *
import numpy as np
import matplotlib.pyplot as plt


r = []
count = 0

## MULTIPROCESSING WITH OR WITHOUT PYGAME ###############################################
def eval_genome(genome, config):
    global r, count

    r.append(compute_R(genome, random=True)[0])
    count+=1
    print count
    
    if count == 1000:
        print r
        print "R_mean: " + str(np.mean(r))
        print "R_std: " + str(np.std(r))
        h, b = np.histogram(r, density=True)
        binWidth = b[1] - b[0]
        plt.bar(b[:-1], h * binWidth, binWidth, color='black')
        plt.title('R in random neural networks')
        plt.xlabel('Bits')
        plt.ylabel('p(R)')
        plt.savefig('RandomAgents.png', bbox_inches='tight')
        #plt.show()

    return [0, 0]


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    #p.add_reporter(neat.StdOutReporter(True))

    # Run until a solution is found, or 200 generations are reached.
    pe = neat.ParallelEvaluator(1, eval_genome)
    winner = p.run(pe.evaluate, n=1)


if __name__ == '__main__':
    global r
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-random')
    run(config_path)

