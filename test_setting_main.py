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

from carla_env import CarlaEnv
import pygame

def main(num_runs):
    
    RENDER = True
    TEST_SETTINGS = True
    MAX_STEPS_PER_EPISODE = 1000
    
    try:
        quit_flag = False
        pygame.init()
        pygame.font.init()

        # create in
        env = CarlaEnv()

        max_steps_per_episode = MAX_STEPS_PER_EPISODE

        clock = pygame.time.Clock()


        for _ in range(num_runs):

            state = env.reset()
            episode_reward = 0
            for timestep in range(max_steps_per_episode):
                clock.tick_busy_loop(60)

                # check quit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_flag = True

                if TEST_SETTINGS:
                    rl_actions = None

                state, reward, done, _ = env.step(rl_actions)
                episode_reward += reward
                env.world.tick(clock)
                if done:
                    break

                if quit_flag:
                    print("stop in the middle ... ")
                    return
            
            print("done in : ", timestep, " -- episode reward: ", episode_reward)
            time.sleep(0.01)
    finally:

        if env.world is not None:
            env.world.destroy()

        pygame.quit()

if __name__ == '__main__':
    main(num_runs = 10)
    
    
    