#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Jiqian's tests

"""

import time
import os
import numpy as np
import sys
import glob

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import pygame

from utils import World, HUD
import gym



class CarlaEnv(object):
    '''
        An OpenAI Gym Environment for CARLA.
    '''

    def __init__(self,
                 host='127.0.0.1',
                 port=2000,
                 city_name='Town03'):
        self.client = carla.Client(host,port)
        self.client.set_timeout(2.0)


        self.hud = HUD(1700,1000)
        self.world = World(self.client.get_world(), self.hud)
        # self.world = World(self.client.load_world(city_name), self.hud)
                

    @staticmethod
    def action_space(self):
        pass

    @staticmethod
    def state_space(self):
        pass

    def reset(self):
        self.world.destroy()
        self.world.restart()


    def step(self,rl_actions):
        throttle = rl_actions['throttle']
        





def main(num_runs):
    quit_flag = False
    try:
        pygame.init()
        pygame.font.init()
        env = CarlaEnv()

        max_steps = 1000


        display = pygame.display.set_mode(
            (1700, 1000),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        clock = pygame.time.Clock()
        for _ in range(num_runs):

            for timestep in range(max_steps):
                clock.tick_busy_loop(60)


                # check quit
                # for event in pygame.event.get():
                #     if event.type == pygame.QUIT:
                #         quit_flag = True

                env.world.tick(clock)


                env.world.render(display)
                pygame.display.flip()

                if quit_flag:
                    return
            print("done in : ", timestep)
            env.reset()

    finally:

        if env.world is not None:
            env.world.destroy()

        pygame.quit()



if __name__ == '__main__':
    main(num_runs = 3)