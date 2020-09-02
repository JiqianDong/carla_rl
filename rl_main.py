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
import pandas as pd 


def gather_data(env, num_runs, max_steps_per_episode, save_info=False):
    CAV_infos = []
    HDV_infos = []
    clock = pygame.time.Clock()
    quit_flag = False
    for episode in range(num_runs):
        state = env.reset()
        episode_reward = 0
        for timestep in range(max_steps_per_episode):
            clock.tick()
            env.world.tick(clock)
            # check quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_flag = True
            rl_actions = np.random.choice(3,2)
            state, reward, done, _ = env.step(rl_actions) #state: {"CAV":[window_size, num_features=9], "LHDV":[window_size, num_features=6]}
            # print(np.array(state['CAV']).shape) # (5,9)
            episode_reward += reward

            if done:
                break

            if quit_flag:
                print("stop in the middle ... ")
                return
        
        print("done in : ", timestep, " -- episode reward: ", episode_reward)

    if save_info:
        CAV_info = pd.DataFrame(CAV_infos,columns=['veh_id','episode','episode_step','px','py','sx','sy','ax','ay','throttle','steer','brake'])
        CAV_info.to_csv('./experience_data/CAV_info.csv',index=False)

        HDV_info = pd.DataFrame(HDV_infos,columns=['veh_id','episode','episode_step','px','py','sx','sy','ax','ay'])
        HDV_info.to_csv('./experience_data/HDV_info.csv',index=False)

def main(num_runs):
    env = None
    RENDER = True
    MAX_STEPS_PER_EPISODE = 300
    WARMING_UP_STEPS = 50
    WINDOW_SIZE = 5
    SAVE_INFO = False

    GATHER_DATA = True
    
    try:
        
        pygame.init()
        pygame.font.init()

        # create environment
        env = CarlaEnv(render_pygame=RENDER,warming_up_steps=WARMING_UP_STEPS,window_size=WINDOW_SIZE)
        
        # 

        max_steps_per_episode = MAX_STEPS_PER_EPISODE

        if GATHER_DATA:
            gather_data(env, num_runs=10, max_steps_per_episode=max_steps_per_episode, save_info=False)

            # time.sleep(0.01)
    finally:


        if env and env.world is not None:
            env.world.destroy()
            # env.world.destroy_all_actors()
            # env.sych_distroy()
            
            settings = env._carla_world.get_settings()
            settings.synchronous_mode = False
            env._carla_world.apply_settings(settings)
            print('\ndisabling synchronous mode.')

        pygame.quit()

if __name__ == '__main__':
    main(num_runs = 10)
    
    
    