#!/usr/bin/env python

import os
import sys
sys.path.append('./_utils/')
sys.path.append('./neat')
import __init__ as neat
sys.modules["neat"] = neat
import visualize
import pickle
from R import *
from ExtraReporter import *


pygame = False
num_trials = 3
n_rewards = 0
field_size = 5.


## MULTIPROCESSING WITH OR WITHOUT PYGAME ###############################################
def eval_genome(genome, config):
    res = compute_R(genome)
    return res[-1], res[0]



def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    ExtraStats = ExtraReporter()
    p.add_reporter(stats)
    p.add_reporter(ExtraStats)
    p.add_reporter(neat.StdOutReporter(True))

    # Run until a solution is found, or 200 generations are reached.
    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = p.run(pe.evaluate, n=100)
    print "This is the R for the winner: " + str(compute_R(winner)[:2])

    # Save the winner.
    with open('winner', 'wb') as f:
        pickle.dump(winner, f)

    # Display the winning genome.
    print 'Winner:'
    print winner

### VISUALIZE STATS AND NETWORK ############################################################
    visualize.plot_stats(stats, ylog=True, view=True, filename="fitness.svg")
    visualize.plot_ExtraStats(ExtraStats, ylog=True, view=True, filename="R_evolution.svg")
    visualize.plot_species(stats, view=True, filename="speciation.svg")

    node_names = {-1: 'W1', -2: 'W2', -3: 'Ep1', -4: 'Ep2', -5: 'Ep3', -6: 'Ep4', -7: 'Ep5', 0: 'ML', 1: 'MR'}
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-nn.gv", show_disabled=False, prune_unused=True)




if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-nn')
    run(config_path)

