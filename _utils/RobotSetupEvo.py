import numpy as np
import math
import Box2DWorld
from Box2DWorld import (createBox, createCircle, collisions)

from VectorFigUtils import vnorm, dist
import random

from Agents import (EvoAgent, Predator, Prey)

def addStandardWall(pos,L,W,name,angle=0):
    createBox(pos, w=W, h=L, bDynamic=False, damping=-1, name=name,angle=angle)


def addReward(who, pos=(0,0), vel=(0,0), bDynamic=False, bCollideNoOne=False):
    name = "reward"
    obj = createCircle(position=pos, bDynamic=bDynamic, bCollideNoOne=bCollideNoOne, density=5, damping=0, friction=0, name=name, r=.2, maskBits=0x0019)
    obj.userData["RGB"] = [0,255,0] # ideally add some noise
    obj.userData["ignore"] = 1.0
    obj.userData["visible"] = True
    obj.userData["name"] = name
    obj.userData["worth"] = 1. #rew  #all rewards equally energetic, for now
    obj.userData['chem'] = [1.,1.,1.,1.] # ideally add some noise and dependence on energy worth
    obj.linearVelocity = vel
    who.objs.append(obj)


# *****************************************************************
# Experimental Setup Epuck Preys
# *****************************************************************

class Setup(object):
    """Exp setup class with two epucks and two reward sites."""

    def __init__(self, net, j, n_rewards=0, type="prey", ghostMode=False, n=2, debug=False, n_obstacles=0, n_visual_sensors=10,field_size=5.):
        """Create the two epucks, two rewards and walls."""
        global bDebug

        bDebug = debug
        print ("-------------------------------------------------")

        self.stop = False
        L_wall=field_size
        self.L_wall=L_wall
        W_wall=.1
        positions=[]
        angles=[2*np.pi, np.pi]
        for _ in range(n):
            #x=1+int(random.random()*4.)/4.*2*L_wall*.8-L_wall*.8
            #y=1+int(random.random()*4.)/4.*2*L_wall*.8-L_wall*.8
            #positions.append([x,y]*np.random.normal(1, .2, 2))
            angles.append(random.random()*2 * np.pi)

        pos = [(field_size / 2., -field_size / 2.),
               (field_size / 2., field_size / 2.),
               (field_size / 2., 0.),
               #(field_size / math.sqrt(2.), 0.),
               (field_size / 2., -field_size / 2.),
               (field_size / 2., field_size / 2.),
               (field_size / 2., 0.)]
               #(field_size / math.sqrt(2.), 0.)]
               #(-field_size / 2., -field_size / 2.),
               #(-field_size / 2., field_size / 2.)]

        #rr = np.random.choice(4, 2, replace=False)
        self.epucks = [EvoAgent(net=net, position=(-field_size/2., 0), angle=angles[0], frontIR=4, nother=1, nrewsensors=2, nvs=n_visual_sensors)]
        if type == "predator":
            self.epucks.append(Predator(position=pos[j], angle=angles[1], frontIR=4, nother=1, nrewsensors=2, nvs=n_visual_sensors))
        elif type == "prey":
            self.epucks.append(Prey(position=pos[j], angle=angles[1], frontIR=4, nother=1, nrewsensors=2, nvs=n_visual_sensors))


        addStandardWall((-L_wall, 0), L_wall, W_wall, 'wall_W')
        addStandardWall((L_wall, 0), L_wall, W_wall, 'wall_E')
        addStandardWall((0, L_wall), W_wall, L_wall, 'wall_N')
        addStandardWall((0, -L_wall), W_wall, L_wall, 'wall_S')

        for obs in range(n_obstacles):
            x = 1 + int(random.random() * 5.) / 5. * 2 * L_wall - L_wall
            y = 1 + int(random.random() * 5.) / 5. * 2 * L_wall - L_wall
            w = .2
            l = int(random.random() * 5.) / 5. * 2
            ang = int(random.random() > .5) * np.pi / 2.  # *2*np.pi
            addStandardWall((x, y), w, l, 'wall_' + str(obs).zfill(2), angle=ang)

        self.objs = []
        for _ in range(n_rewards):
            addReward(self, pos=(random.uniform(-1, 1) * self.L_wall, random.uniform(-1, 1) * self.L_wall),
                        vel=(0, 0), bDynamic=True, bCollideNoOne=False)


    def update(self,i):
        # update rewards: exponential decay for their value ( at the end of the 400 steps, final value ~= 1/3 )
        for r in self.objs:
            r.userData["worth"] = np.exp(-.0025*i)

        # update epucks
        for ii, e in enumerate(self.epucks):

            if i == 3:  # avoid spurious consumption of food in the beginning of experiment
                e.food = 0

            e.update()
            pos = e.getPosition()
            #e.motors = [0,0]

            for g in e.GradSensors:
                centers = [[o.position, o.userData['name']] for o in self.objs]
                if len(self.epucks)>1:
                    centers += [[o.getPosition(), o.userData['name']] for o in [x for k,x in enumerate(self.epucks) if k!=ii]]
                g.update(pos, e.getAngle(), centers)

        if dist(self.epucks[0].getPosition(), self.epucks[1].getPosition()) < self.epucks[0].r*2 + .1:
            self.stop=True


    def reset(self):
        for o in self.objs:
            o.userData["name"] += "_destroy"

        for e in self.epucks:
            Box2DWorld.TODESTROY.append(e.body)
            e.body.userData['name'] += '_destroy'
