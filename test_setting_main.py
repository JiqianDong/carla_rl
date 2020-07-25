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
    env = None
    RENDER = False
    MAX_STEPS_PER_EPISODE = 30
    
    try:
        quit_flag = False
        pygame.init()
        pygame.font.init()

        # create in
        env = CarlaEnv(render_pygame=RENDER)

        max_steps_per_episode = MAX_STEPS_PER_EPISODE

        clock = pygame.time.Clock()


        for _ in range(num_runs):

            state = env.reset()
            episode_reward = 0
            for timestep in range(max_steps_per_episode):
                clock.tick()
                env.world.tick(clock)
                env._carla_world.tick()

                # check quit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_flag = True

                rl_actions = np.random.choice(3,2)

                state, reward, done, _ = env.step(rl_actions)
                print("current control", env.world.cav_controller.current_control)



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

        if env and env.world is not None:
            # env.world.destroy()
            # env.world.destroy_all_actors()
            env.sych_distroy()
            print('\ndisabling synchronous mode.')
            settings = env._carla_world.get_settings()
            settings.synchronous_mode = False
            env._carla_world.apply_settings(settings)
        

        pygame.quit()

if __name__ == '__main__':
    main(num_runs = 10)
    
    
    