import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import json

import json
import matplotlib.pyplot as plt
import numpy as np

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def plot_training(logdir,loss_smooth_weight=0.3,reward_smooth_weight=0.85,loss_y_lim=None, reward_y_lim=None):
    import glob
    data_files = glob.glob(logdir+'*training_hist.txt')
    losses = []
    losses_smoothed = []
    rewards = []
    rewards_smoothed = []
    names = []
    data_files.sort()
    for data_file in data_files:

        with open(data_file,'rb') as f:
            data = json.load(f)
        names.append(data_file.split('/')[-1].split('_')[0])


        loss = np.array(data['loss'])
        reward = np.array(data['episode_reward'])
        loss = loss[~np.isnan(loss)] #drop nan for smoothing

        loss_smoothed = smooth(loss, loss_smooth_weight)
        reward_smoothed = smooth(reward,reward_smooth_weight)

        losses.append(loss)
        rewards.append(reward)

        losses_smoothed.append(loss_smoothed)
        rewards_smoothed.append(reward_smoothed)


    plt.figure()
    for (loss,loss_smoothed,name) in zip(losses,losses_smoothed,names):
        p = plt.plot(np.arange(len(loss)),loss,alpha=0.2)
        plt.plot(np.arange(len(loss_smoothed)),loss_smoothed,label=name,c=p[0].get_color())
    plt.legend()
    if loss_y_lim:
        plt.ylim(loss_y_lim)
    plt.title('Loss vs episode')
    plt.savefig('./figures/loss.png',dpi=300)

    plt.figure()

    for (reward,reward_smoothed,name) in zip(rewards,rewards_smoothed,names):

        p = plt.plot(np.arange(len(reward)),reward,alpha=0.2)

        plt.plot(np.arange(len(reward_smoothed)),reward_smoothed,label=name,c=p[0].get_color())

    plt.plot([0,len(reward_smoothed)-1],[-6281.482498+2000]*2,label="rule_based")

    plt.legend()
    if reward_y_lim:
        plt.ylim(reward_y_lim)
    plt.title('rewards vs episode')
    plt.savefig('./figures/rewards.png',dpi=300)



def loss_plot(logdir):
    import glob
    data_files = glob.glob(logdir+'*.txt')
    training_losses = []
    model_names = []
    # valid_losses = []
    data_files.sort()

    for data_file in data_files:
        with open(data_file,'r') as f:
            data = json.load(f)
        model_names.append(data_file.split('/')[-1].split('_')[0])

        training_loss = data['training_loss']
        training_losses.append(training_loss)
        # valid_loss = data['validation_loss']
    plt.figure()
    for model_name, loss in zip(model_names, training_losses):
        plt.plot(loss,label=model_name)

    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.title('CAV trajectory prediction loss training curve')
    plt.legend()
    plt.savefig('./figures/cav training curve.png',dpi=300)

if __name__ == '__main__':
    import glob
    plot_training('./logs/',0.3,0.9,(10,70),(-10000,3000))
    #plot_training('./logs/',0.3,0.9)