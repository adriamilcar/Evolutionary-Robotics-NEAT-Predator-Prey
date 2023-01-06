#!/usr/bin/env python

import sys
sys.path.append('./_utils/')
import numpy as np
import pygame
import pygame.surfarray as surfarray
from pygame.locals import *
import PyGameUtils
import Box2DWorld 
from RobotSetupEvo import Setup


def getPPM(field_size=5., screenSize=640):
    def_field_size = 5.
    scale = field_size / def_field_size
    ppm_default = screenSize * 65. / 640.
    return int(ppm_default / scale)


class exp_rewards(object):
    def __init__(self, net, j, n_rew=4, type="prey", f_size=4.):
        field_size = f_size
        screenSize = 500
        self.type = type

        ppm = getPPM(field_size, screenSize)
        pygame.init()

        PyGameUtils.setScreenSize(screenSize, screenSize, ppm=ppm, center=True)
        box2dWH = (PyGameUtils.SCREEN_WIDTH, PyGameUtils.SCREEN_HEIGHT)

        flags = HWSURFACE | DOUBLEBUF | RESIZABLE
        self.screen = pygame.display.set_mode(box2dWH, flags, 32)
        self.screen.set_alpha(None)
        surfarray.use_arraytype('numpy')

        pygame.display.set_caption('Epuck Simulation')
        self.clock = pygame.time.Clock()

        self.exp = Setup(net=net, j=j, n_rewards=n_rew, type=type, debug=True, field_size=field_size)

    def run(self):
        i = 0
        running = True

        while running and i<300 and not self.exp.stop:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: #or event.key == pygame.K_ESCAPE:
                        # The user closed the window or pressed escape
                        running = False

                self.screen.fill((0, 0, 0, 0))
                flag = Box2DWorld.destroy([self.exp.objs, self.exp.epucks])
                PyGameUtils.draw_world(self.screen)

                self.exp.update(i)

                Box2DWorld.step()

                pygame.display.flip()  # Flip the screen and try to keep at the target FPS
                self.clock.tick(Box2DWorld.TARGET_FPS)
                pygame.display.set_caption("FPS: {:6.3}{}".format(self.clock.get_fps(), " " * 5))

                i += 1

            except KeyboardInterrupt:
                print ('error 0')
                break

        fitness = self.exp.epucks[0].time
        pos = [e.pos for e in self.exp.epucks]
        outputs = self.exp.epucks[0].acc_outputs
        self.exp.reset()
        flag = Box2DWorld.destroy([self.exp.objs, self.exp.epucks])
        pygame.display.quit()
        pygame.quit()
        print('Done!')
        return [fitness, pos, outputs]

