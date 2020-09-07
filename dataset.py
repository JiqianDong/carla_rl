import numpy as np
from collections import defaultdict
import pickle

class Dataset(object):

    def __init__(self):
        self._states = []
        self._actions = []
        self._next_states = []
        self._rewards = []
        self._dones = []
        self._state_history = defaultdict(list) # for computing state mean, std, 
        self._state_diff_history = defaultdict(list) # for computing state difference mean, std

    def add(self, state, action, next_state, reward, done):
        """
        Add (s, a, r, s') to this dataset
        """

        self._states.append(state)
        self._actions.append(action)
        self._next_states.append(next_state)
        self._rewards.append(reward)
        self._dones.append(done)

        for key,val in state.items():
            self._state_history[key].append(np.array(val))

            if key in next_state:
                self._state_diff_history[key].append(np.array(next_state[key]) - np.array(val))

    def append(self, other_dataset):
        """
        Append other_dataset to this dataset
        """
        self._states += other_dataset._states
        self._actions += other_dataset._actions
        self._next_states += other_dataset._next_states
        self._rewards += other_dataset._rewards
        self._dones += other_dataset._dones

        
    @property
    def is_empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._states)


    @property
    def state_mean(self):
        mean = {}
        for key,val in self._state_history.items():
            mean[key] = np.mean(val,axis=0) 
        return mean


    @property
    def state_std(self):
        std = {}
        for key,val in self._state_history.items():
            std[key] = np.std(val,axis=0) 
        return std


    @property
    def delta_state_mean(self):
        mean = {}
        for key,val in self._state_diff_history.items():
            mean[key] = np.mean(val,axis=0) 
        return mean

    @property
    def delta_state_std(self):
        std = {}
        for key,val in self._state_diff_history.items():
            std[key] = np.std(val,axis=0) 
        return std


    # def rollout_iterator(self):
    #     """
    #     Iterate through all the rollouts in the dataset sequentially
    #     """
    #     end_indices = np.nonzero(self._dones)[0] + 1

    #     states = np.asarray(self._states)
    #     actions = np.asarray(self._actions)
    #     next_states = np.asarray(self._next_states)
    #     rewards = np.asarray(self._rewards)
    #     dones = np.asarray(self._dones)

    #     start_idx = 0
    #     for end_idx in end_indices:
    #         indices = np.arange(start_idx, end_idx)
    #         yield states[indices], actions[indices], next_states[indices], rewards[indices], dones[indices]
    #         start_idx = end_idx

    def random_iterator(self, batch_size, return_sequence=True):
        """
        Iterate once through all (s, a, r, s') in batches in a random order
        For only training the system dynamic function only.
        """
        all_indices = np.nonzero(np.logical_not(self._dones))[0]
        np.random.shuffle(all_indices)
        actions = np.array(self._actions)

        i = 0
        while i < len(all_indices):
            indices = all_indices[i:i+batch_size]
            output_state = defaultdict(list)
            output_next_state = defaultdict(list)
            for ind in indices:
                current = self._states[ind]
                next_ = self._next_states[ind]
                for key, val in current.items():
                    output_state[key].append(val)

                for key,val in next_.items():
                    if return_sequence:
                        output_next_state[key].append(val)
                    else:
                        output_next_state[key].append(val[-1,:])

            for key in output_state.keys():
                output_state[key] = np.array(output_state[key])

            for key in output_next_state.keys():
                output_next_state[key] = np.array(output_next_state[key])
            
            yield output_state, actions[indices], output_next_state

            i += batch_size

