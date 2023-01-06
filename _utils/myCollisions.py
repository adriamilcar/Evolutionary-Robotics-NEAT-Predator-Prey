from Box2D import b2ContactListener

TODESTROY = []


class consumeReward(b2ContactListener):
    def __init__(self):
        b2ContactListener.__init__(self)
        print 'started'

    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass

    def PreSolve(self, contact, oldManifold):
        global TODESTROY
        worldManifold = contact.worldManifold
        bA = contact.fixtureA.body
        bB = contact.fixtureB.body
        bodies = [bA, bB]
        names = [bA.userData['name'], bB.userData['name']]
        re = False
        ep = False
        for i, name in enumerate(names):
            if 'reward' in name:
                re = True
                destroy = i
            if 'epuck' in name:
                ep = True
        if re and ep:
            TODESTROY.append(bodies[destroy])
            bodies[1 - destroy].userData['food'].append(0 + bodies[destroy].userData['worth'])
            bodies[destroy].awake = False
            bodies[destroy].userData['visible'] = False
            bodies[destroy].userData['name'] += '_destroy'
            print bodies[destroy]
            print 'Collision detected'

    def PostSolve(self, contact, impulse):
        pass


class cl(b2ContactListener):
    pass


class wallsDestroy(b2ContactListener):
    # Sacha's wall-collision destroyer
    def __init__(self):
        b2ContactListener.__init__(self)

    def BeginContact(self, contact):
        global TODESTROY
        bA, bB = contact.fixtureA.body, contact.fixtureB.body
        for b in [bA, bB]:
            if b.userData:
                if 'epuck' in b.userData['name']:
                    TODESTROY.append(b)

    def EndContact(self, contact):
        pass

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass


class collisionDestruction(b2ContactListener):
    '''
    To use, Add to the world loading main file:
    try:
        imp.find_module('myCollisions')
        found = True
    except ImportError:
        found = False
    if found:
        from myCollisions import TODESTROY, collisionDestruction
        world.contactListener = collisionDestruction([[objectToDestroy,ObjectDestructor],[anotherPair,OfDestructionObjects],['reward','epuck'],['epuck', 'wall']]) # For example
    else:
        TODESTROY=[]

    '''

    def __init__(self, dicts=[['reward', 'prey'], ['prey', 'predator']], sounds=False):  # ,['epuck','wall']]):
        b2ContactListener.__init__(self)
        self.dicts = dicts
        print 'started'
        self.sounds = sounds

    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass

    def PreSolve(self, contact, oldManifold):
        global TODESTROY
        worldManifold = contact.worldManifold
        bA = contact.fixtureA.body
        bB = contact.fixtureB.body
        bodies = [bA, bB]
        names = [bA.userData['name'], bB.userData['name']]

        for dic in self.dicts:
            ep = False
            re = False
            for i, name in enumerate(names):
                if dic[0] in name:
                    re = True
                    destroy = i
                if dic[1] in name:
                    ep = True
                if re and ep:

                    TODESTROY.append(bodies[destroy])
                    if 'food' in bodies[destroy].userData.keys():
                        print 'destroying an epuck!'
                    if 'worth' in bodies[destroy].userData.keys():
                        print bodies[destroy].userData['name'], bodies[1 - destroy].userData['name']

                        bodies[1 - destroy].userData['food'].append(0 + bodies[destroy].userData['worth'])
                    bodies[destroy].awake = False
                    bodies[destroy].userData['visible'] = False
                    bodies[destroy].userData['name'] += '_destroy'
                    print 'Collision detected'

    def PostSolve(self, contact, impulse):
        pass
