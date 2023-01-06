from Robots import *
import os
import neat
import pickle


class GradSensorPlus(object):
    """ Gradient Sensor used by EPuck."""

    def __init__(self, ngrad=1, name="grad", maxdist=3.):
        """Init Gradient Sensor."""
        self.name = name  # name has to be the kind of chemical it detects

        self.ngrad = ngrad
        self.maxd = maxdist
        if (ngrad < 4):
            m, da = (1 + ngrad) % 2, np.pi / (2 + ngrad)
        elif (ngrad == 4):
            m, da = (1 + ngrad) % 2, np.pi / ngrad
        else:
            m, da = (1 + ngrad) % 2, np.pi / (ngrad - 1)
        self.GradAngles = [k * da - ((ngrad - m) / 2) * da - m * da / 2 for k in range(ngrad)]
        self.GradValues = [0 for i in range(ngrad)]

    def update(self, pos, angle, centers=[], extremes=0):
        """Update passing agnet pos, angle and list of positions of gradient emmiters."""
        sensors = range(self.ngrad)
        if extremes:
            sensors = [0, self.ngrad - 1]

        if len(centers) == 0: return
        for k in sensors:
            v = vrotate((1, 0), angle + self.GradAngles[k])
            vals = [0 for i in range(len(centers))]
            for i, cl in enumerate(centers):
                c = cl[0]
                vc = (c[0] - pos[0], c[1] - pos[1])
                d = dist(pos, c)
                if d > self.maxd:
                    d = self.maxd
                if cl[1] == self.name:
                    vals[i] += ((self.maxd - d) / self.maxd) * (1 - abs(vangle(v, vc)) / np.pi)
            self.GradValues[k] = max(vals)



class defaultAgent(Epuck):
    def __init__(self, position=(0, 0), angle=np.pi / 2, r=0.3, bHorizontal=False, frontIR=4, nother=0, nrewsensors=2,
                 RGB=(200, 20, 50), nvs=2, bodyType='circle', categoryBits=0x0001, name='epuck_prey', maskBits=0x0009):
        Epuck.__init__(self, position=position, angle=angle, nother=nother, nrewsensors=nrewsensors, r=r,
                       frontIR=frontIR, categoryBits=categoryBits, name=name, maskBits=maskBits)
        self.VS = VisualSensor(nvs, 180)
        self.energy = 1.
        self.energy_decay = 0  # This is a parameter to tune
        self.body.userData['food'] = []
        self.userData["RGB"] = RGB
        self.pos = [np.array(position) + 5.]


    def updateVariables(self):

        #self.energy = np.clip(self.energy-self.energy_decay-abs(self.motors[0]**2+self.motors[1]**2)*.0001, 0,1)

        if len(self.body.userData['food']) > 0:
            self.energy += (1. - self.energy) * self.body.userData['food'].pop()

        Epuck.update(self)
        self.VS.update(self.body.position, self.body.angle, self.r)
        self.pos = np.vstack((self.pos, np.array(self.body.position) + 5.))

    def update(self):
        self.updateVariables()



class EvoAgent(defaultAgent):
    def __init__(self, net, ghostMode=False, position=(0, 0), angle=np.pi / 2, r=0.46, bHorizontal=False,
                 frontIR=4, nother=0, nrewsensors=2, RGB=(255, 0, 0), nvs=2, bodyType='circle'):

        catBits = 0x0001
        if ghostMode:
            catBits = 0x0010

        defaultAgent.__init__(self, position=position, angle=angle, r=r, bHorizontal=bHorizontal, frontIR=frontIR,
                              nother=nother, nrewsensors=nrewsensors, RGB=RGB, nvs=nvs, bodyType=bodyType,
                              categoryBits=catBits)
        #if ghostMode:
        self.body.userData["ignore"] = 1.0

        self.GradSensors = []
        #self.GradSensors.append(GradSensorPlus(ngrad=4, name="reward", maxdist=3.))
        self.GradSensors.append(GradSensorPlus(ngrad=4, name="epuck_prey", maxdist=8.))

        self.motors = [0., 0.]
        self.acc_outputs = [0., 0., 0.]

        self.time = 0.  # lifetime
        self.food = 0  # amount of accumulated food through lifetime
        self.dis = []   # accumulated distances to the other epuck

        self.net = net


    def updateVariables(self):
        if len(self.body.userData['food']) > 0:
            self.food += self.body.userData['food'][-1]

        super(EvoAgent, self).updateVariables()

        Gs = []
        for ii in range(len(self.GradSensors)):
            for jj in range(len(self.GradSensors[ii].GradValues)):
                Gs.append(self.GradSensors[ii].GradValues[jj])

        sensors = 1 - np.array(self.IR.IRValues)
        sensors = np.append(sensors, Gs)
        #self.motors = (np.array(self.net.advance(sensors, .01, .01)) - .5 )*2  # scale [0, 1] to [-1, 1]
        self.motors = np.array(self.net.advance(sensors, .01, .01))

        #self.acc_motors = np.vstack((self.acc_motors, self.motors))
        self.acc_outputs = np.vstack((self.acc_outputs, [v for (k,v) in self.net.values[1-self.net.active].items() if k>=0 and k!=406]))

    def update(self):
        self.time += 1.
        self.updateVariables()



class Predator(defaultAgent):
    def create_net(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-predator')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        with open('_utils/predator', 'rb') as f:
            gen = pickle.load(f)
        return neat.ctrnn.CTRNN.create(gen, config, .01)


    def __init__(self, type="slow", ghostMode=False, position=(0, 0), angle=np.pi / 2, r=0.46, bHorizontal=False,
                 frontIR=4, nother=0, nrewsensors=2, RGB=(0, 0, 255), nvs=2, bodyType='circle'):

        catBits = 0x0001
        if ghostMode:
            catBits = 0x0010

        defaultAgent.__init__(self, position=position, angle=angle, r=r, bHorizontal=bHorizontal, frontIR=frontIR,
                              nother=nother, nrewsensors=nrewsensors, RGB=RGB, nvs=nvs, bodyType=bodyType,
                              categoryBits=catBits)
        #if ghostMode:
        self.body.userData["ignore"] = 1.0

        self.GradSensors = []
        self.GradSensors.append(GradSensorPlus(ngrad=4, name="epuck_prey", maxdist=8.))

        self.motors = [0., 0.]

        self.time = 0  # lifetime
        self.food = 0  # amount of accumulated food through lifetime

        self.net = self.create_net()

        if type=="slow":
            self.scale = 2
        elif type=="fast":
            self.scale = 3


    def updateVariables(self):
        if len(self.body.userData['food']) > 0:
            self.food += self.body.userData['food'][-1]

        super(Predator, self).updateVariables()

        Gs = []
        for ii in range(len(self.GradSensors)):
            for jj in range(len(self.GradSensors[ii].GradValues)):
                Gs.append(self.GradSensors[ii].GradValues[jj])

        sensors = 1 - np.array(self.IR.IRValues)
        sensors = np.append(sensors, Gs)
        self.motors = (np.array(self.net.advance(sensors, .01, .01)) - .5 )*self.scale # scale [0, 1] to [-X, X] (X=1 if scale==2)


    def update(self):
        self.time += 1
        self.updateVariables()





class Prey(defaultAgent):
    def create_net(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-prey')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        with open('_utils/prey', 'rb') as f:
            gen = pickle.load(f)
        return neat.ctrnn.CTRNN.create(gen, config, .01)


    def __init__(self, type="slow", ghostMode=False, position=(0, 0), angle=np.pi / 2, r=0.46, bHorizontal=False,
                 frontIR=4, nother=0, nrewsensors=2, RGB=(0, 255, 0), nvs=2, bodyType='circle'):

        catBits = 0x0001
        if ghostMode:
            catBits = 0x0010

        defaultAgent.__init__(self, position=position, angle=angle, r=r, bHorizontal=bHorizontal, frontIR=frontIR,
                              nother=nother, nrewsensors=nrewsensors, RGB=RGB, nvs=nvs, bodyType=bodyType,
                              categoryBits=catBits)
        #if ghostMode:
        self.body.userData["ignore"] = 1.0

        self.GradSensors = []
        self.GradSensors.append(GradSensorPlus(ngrad=4, name="epuck_prey", maxdist=3.))

        self.motors = [0., 0.]

        self.time = 0  # lifetime
        self.food = 0  # amount of accumulated food through lifetime

        self.net = self.create_net()

        if type=="slow":
            self.scale = 2
        elif type=="fast":
            self.scale = 3


    def updateVariables(self):
        if len(self.body.userData['food']) > 0:
            self.food += self.body.userData['food'][-1]

        super(Prey, self).updateVariables()

        Gs = []
        for ii in range(len(self.GradSensors)):
            for jj in range(len(self.GradSensors[ii].GradValues)):
                Gs.append(self.GradSensors[ii].GradValues[jj])

        sensors = 1 - np.array(self.IR.IRValues)
        sensors = np.append(sensors, Gs)
        self.motors = (np.array(self.net.advance(sensors, .01, .01)) - .5 )*self.scale # scale [0, 1] to [-X, X] (X=1 if scale==2)


    def update(self):
        self.time += 1
        self.updateVariables()