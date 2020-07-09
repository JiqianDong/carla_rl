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
import numpy as np

import gym
from gym.spaces.box import Box
from gym.spaces import Discrete, Tuple


from utils import World, HUD

class CarlaEnv(object):
    '''
        An OpenAI Gym Environment for CARLA.
    '''

    def __init__(self,
                 host='127.0.0.1',
                 port=2000,
                 city_name='Town03',
                 render_pygame=True):
        self.client = carla.Client(host,port)
        self.client.set_timeout(2.0)

        self.hud = HUD(1700,1000)
        self.world = World(self.client.get_world(), self.hud)
        self.render_pygame = render_pygame

        self.timestep = 0




    @staticmethod
    def action_space(self):
        throttle_brake = Discrete(3)  # -1 brake, 0 keep, 1 throttle
        steering = Discrete(3)
        return Tuple([throttle_brake,steering])

    @staticmethod
    def state_space(self):
        N = len(self.world.vehicles)
        F = 6 # FIXME not hard code
        return Box(low=-np.inf, high=np.inf, shape=(N,F), dtype=np.float32)

    def reset(self):
        # reset the render display panel
        if self.render_pygame:
            self.display = pygame.display.set_mode((1700,1000),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
            
        self.world.destroy()
        self.world.restart()


        time.sleep(0.001)
        self.timestep = 0

        return self.get_state()


    def step(self,rl_actions):
        
        self.world.cav_controller.step(rl_actions)
        self.world.ldhv_controller.step()
        self.world.bhdv_controller.step()

        state = self.get_state() #next observation

        collision = self.check_collision()
        done = False
        if collision:
            print(collision)
            done = True

        reward = self.compute_reward(collision)

        self.timestep += 1 
        infos = {}

        if self.render_pygame:
            self.render_frame()

        return state, reward, done, infos

    def render_frame(self):
        if self.display:
            self.world.render(self.display)
            pygame.display.flip()
        else:
            raise Exception("No display to render")

    def check_collision(self):
        if len(self.world.collision_sensor.history)>0:
            return self.world.collision_sensor.history[-1]
        else:
            return None

    def get_state(self):
        states = []
        for veh in self.world.vehicles:
            state = []
            location = veh.get_location()
            state += [location.x, location.y]

            speed = veh.get_velocity()
            state += [speed.x, speed.y]

            accel = veh.get_acceleration()
            state += [accel.x, accel.y]
            states.append(np.array(state))
        return np.array(states)

    def compute_reward(self,collision=None):

        weight_collision = 1
        base_reward = 0


        collision_penalty = 0
        if collision:
            collision_penalty = collision[1] # the negative intensity of collision

        return base_reward - collision_penalty*weight_collision
