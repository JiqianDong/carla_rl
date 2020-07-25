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

def main(num_runs):
    env = None
    RENDER = True
    MAX_STEPS_PER_EPISODE = 300
    SAVE_INFO = True
    
    try:
        quit_flag = False
        pygame.init()
        pygame.font.init()

        # create in
        env = CarlaEnv(render_pygame=RENDER)

        max_steps_per_episode = MAX_STEPS_PER_EPISODE

        clock = pygame.time.Clock()

        CAV_infos = []
        HDV_infos = []

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

                state, reward, done, _ = env.step(rl_actions)

                # print(state)
                cav_control = env.world.cav_controller.current_control


                # print("current control: ", cav_control)
                # print("carla control: ", env.world.CAV.get_control())
                # print()
                for veh_id, state_vals in state.items():

                    if veh_id == 'CAV':
                        CAV_info = [veh_id,episode,timestep] +state_vals+ list(cav_control.values())

                        CAV_infos.append(CAV_info)
                    else:
                        hdv_info = [veh_id,episode,timestep] + state_vals
                        HDV_infos.append(hdv_info)

                episode_reward += reward


                if done:
                    break

                if quit_flag:
                    print("stop in the middle ... ")
                    return
            
            print("done in : ", timestep, " -- episode reward: ", episode_reward)
            # time.sleep(0.01)
    finally:
        if SAVE_INFO:
            cav_info = pd.DataFrame(CAV_infos,columns=['veh_id','episode','episode_step','px','py','sx','sy','ax','ay','throttle','steer','brake'])
            cav_info.to_csv('./experience_data/cav_info.csv',index=False)

            hdv_info = pd.DataFrame(HDV_infos,columns=['veh_id','episode','episode_step','px','py','sx','sy','ax','ay'])
            hdv_info.to_csv('./experience_data/hdv_info.csv',index=False)

        if env and env.world is not None:
            env.world.destroy()
            # env.world.destroy_all_actors()
            # env.sych_distroy()
            print('\ndisabling synchronous mode.')
            settings = env._carla_world.get_settings()
            settings.synchronous_mode = False
            env._carla_world.apply_settings(settings)
        

        pygame.quit()

if __name__ == '__main__':
    main(num_runs = 10)
    
    
    