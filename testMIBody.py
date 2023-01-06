#!/usr/bin/env python

import sys
sys.path.append('./_utils/')
import pickle
from MI_body import *
import matplotlib.pyplot as plt


## LOAD WINNER GENOME AND COMPUTE R ########################################################
with open('winner', 'rb') as f:
    gen = pickle.load(f)

print gen

i=[]
for max_iter in range(1,301):
    i.append(compute_I(gen, max_iter)[0])

plt.plot(range(300), i)
plt.title("Mutual information between Body Position and other's Identity")
plt.xlabel("Time steps")
plt.ylabel("I (BodyPos ; Identity)")
plt.show()
