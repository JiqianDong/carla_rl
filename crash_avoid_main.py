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
import pickle


def gather_data(env, num_runs, max_steps_per_episode, save_info=False):
    from dataset import Dataset
    dataset = Dataset()
    clock = pygame.time.Clock()
    quit_flag = False
    for episode in range(num_runs):
        current_state = env.reset().copy()

        episode_reward = 0
        for timestep in range(max_steps_per_episode):
            clock.tick()
            env.world.tick(clock)
            # check quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_flag = True
            rl_actions = np.random.choice(3,2)  # -1 brake, 0 keep, 1 throttle,  steering increment (-1,0,1)
            # print(rl_actions)
            next_state, reward, done, _ = env.step(rl_actions) #state: {"CAV":[window_size, num_features=9], "LHDV":[window_size, num_features=6]}
            # print(np.array(state['current_control']).shape) #(5,3) throttle, steering, brake
            # print(env.world.CAV.get_control(),'\n')
            print(current_state, next_state, '\n')
            episode_reward += reward
            dataset.add(current_state,rl_actions,next_state,reward,done)
            current_state = next_state

            if done:
                break

            if quit_flag:
                print("stop in the middle ... ")
                return

        print("done in : ", timestep, " -- episode reward: ", episode_reward)

    if save_info:
        with open('./experience_data/data_pickle.pickle','wb') as f:
            pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)

def main(num_runs):
    env = None
    RENDER = True
    MAX_STEPS_PER_EPISODE = 300
    WARMING_UP_STEPS = 50
    WINDOW_SIZE = 5
    SAVE_INFO = True
    RETURN_SEQUENCE = False

    GATHER_DATA = True
    TRAINING = True
    
    try:
        
        pygame.init()
        pygame.font.init()

        # create environment
        env = CarlaEnv(render_pygame=RENDER,warming_up_steps=WARMING_UP_STEPS,window_size=WINDOW_SIZE)
        
        max_steps_per_episode = MAX_STEPS_PER_EPISODE

        if GATHER_DATA:
            gather_data(env, num_runs=10, max_steps_per_episode=max_steps_per_episode, save_info=SAVE_INFO)

        if TRAINING: 

            pass

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
    
    
    