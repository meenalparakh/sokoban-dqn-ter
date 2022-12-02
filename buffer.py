import numpy as np
import random
from collections import deque

import gym
import torch
import seaborn as sns
from tqdm.notebook import tqdm

from torch import nn
import pandas as pd
from torch import optim
from typing import Any
from copy import deepcopy
# from gym.wrappers import Monitor

from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import random
import time
import os
import gym
import json

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from collections import deque


import operator

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):

        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):

        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        obs_t = obs_t.reshape((-1, 1))
        obs_tp1 = obs_tp1.reshape((-1, 1))
        
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha = 0.6):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def append(self, data):
        self.add(*data)
        
    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta = 0.4):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        state, action, reward, next_state, done = self._encode_sample(idxes)
        return state, action, reward, next_state, done,idxes, weights

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
#             print(priority)
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
            
            
class CyclicBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.cur_pos = 0

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

    def append(self, data):
        #### TODO: add data to the buffer
        #### if the buffer is not full yet, you can simply append the data to the buffer
        #### otherwise, you need to replace the oldest data with the current data (FIFO)
        #### Hint: you may find self.cur_pos useful, it can be used as a position index
        # assert False
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)

        else:
            self.buffer[self.cur_pos] = data
        
        self.cur_pos = (self.cur_pos+1) % self.capacity
        
    def sample(self, batch_size):
        selected_idx = np.random.choice(self.__len__(), batch_size, replace=False)
        array = [self.buffer[i] for i in selected_idx]
        obs= []
        next_obs = []
        actions = []
        rewards =[] 
        dones = []
        for i in range(batch_size):
            obs.append(array[i][0])
            actions.append(array[i][1])
            rewards.append(array[i][2])
            next_obs.append(array[i][3])
            dones.append(array[i][4])
        return np.array(obs), np.array(actions), np.array(rewards), np.array(next_obs), np.array(dones)

    
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done', 'time_step'))    


class ReplayBufferGraph:
    def __init__(self, max_transitions = 1000, vertex_dim = 3, state_dim = 100, projection_matrix = None, verbose = False):
      
        '''
        * unweighted graph of state transitions
        * vertices are projected states
        * edges are transitions (s, a, r, s') = 
                    state, action, reward, next_state
        
        * buffer stores edges, or transitions
        '''
        
        self.verbose = verbose
        self.vertex_dim = vertex_dim
        self.state_dim = state_dim
        
        self.vertices = set()
        self.terminal_vertices = set()
        
        self.buffer = dict()
        self.num_transitions = 0
        self.max_transitions = max_transitions

        if projection_matrix is None:
            projection_matrix = np.random.normal(size = (vertex_dim, state_dim))
        
        self.projection_matrix = projection_matrix
        
        ## variables for constructing queue, currently using a list
        self.search_queue = deque()
        self.batch_queue = deque()
        self.visited_vertices = set()
        
        self.batch_size_list = []
        
        
#         state_ = np.array([19]).reshape((-1, 1))
#         self.TERMINAL_VERTEX = self.projection_matrix @ state_
#         self.TERMINAL_VERTEX  = tuple(self.TERMINAL_VERTEX[0])
        
        
#         self.transitions_based_on_time_step = dict()
        self.time_transitions = dict()
        
    def __len__(self):
        return len(self.time_transitions)

    def __getitem__(self, item):
        return self.buffer[item]
        
    def append(self, data):
        
        ## t_env_step is the environment step when this 
        ##     transition is being added to the environment.
        
        if len(self.time_transitions) == self.max_transitions:
            time_step_to_remove = min(self.time_transitions.keys())
            self.prune_graph(time_step_to_remove)
            
        state, action, reward, next_state, done, t_env_step = data
        
        state = state.reshape((-1, 1))
        next_state = next_state.reshape((-1, 1))

        v = self.projection_matrix @ state
        vd = self.projection_matrix @ next_state
        
        v = tuple(v.T[0])
        vd = tuple(vd.T[0])
    
        if t_env_step not in self.time_transitions:
            self.time_transitions[t_env_step] = Transition(state, action, reward, next_state, done, t_env_step)
        else:
            self.time_transitions[t_env_step].append(Transition(state, action, reward, next_state, done, t_env_step))
            
        if vd not in self.buffer:
            self.vertices.add(vd)
            self.buffer[vd] = dict()

        if v not in self.buffer[vd]:
            self.buffer[vd][v] = dict()
            self.vertices.add(v)
         
        self.buffer[vd][v][t_env_step] = Transition(state, action, reward, next_state, done, t_env_step)
        
        self.num_transitions += 1
        
    def add_to_terminal_vertices(self, next_state):
        next_state = np.array([next_state]).reshape((-1, 1))
        vd = self.projection_matrix @ next_state
        vd = tuple(vd.T[0])
        self.terminal_vertices.add(vd)
        
    def get_predecessor_transitions(self, state):
        
        state = np.reshape(state, (-1, 1))
        
        vd = self.projection_matrix @ state
        vd = tuple(vd.T[0])
        
        if vd not in self.buffer:
            return {}
        return self.buffer[vd]
    
    def step_reverse_BFS(self, n_root_max = 8, n_predecessors_max = 3, transitions_max = 1):

        if self.verbose:
            print(f'Buffer (reverse BFS):')
            self.print_buffer()
            print(f'terminal vertices (reverse BFS): {self.terminal_vertices}')
            print(f'search queue: {self.search_queue}')
            print(f'visited: {self.visited_vertices}')
            
            
        found = False
        
        while not found:
            if (len(self.search_queue) == 0):
                n_root = min(n_root_max, len(self.terminal_vertices))
                if n_root == 0:
                    return False

    #             print('num roots:', n_root)
                ve = random.sample(tuple(self.terminal_vertices), n_root)
                self.search_queue.extend(ve)
                self.visited_vertices = set()
            
            vd = self.search_queue.pop()
            if vd not in self.visited_vertices:
                found = True
            
        if vd in self.buffer:    
            n_predecessors = min(n_predecessors_max, len(self.buffer[vd].keys()))
            predecessors = random.sample(self.buffer[vd].keys(), n_predecessors)

            self.search_queue.extendleft(predecessors)
            for v in predecessors:
                num_transitions = min(transitions_max, len(self.buffer[vd][v]))
                self.batch_queue.extendleft(random.sample(list(self.buffer[vd][v].values()), num_transitions))

        self.visited_vertices.add(vd)

        return True
            
    def print_buffer(self):
        transitions = []
        for vd in self.buffer:
            for v in self.buffer[vd]:
                transitions.append((v, vd))
        print(transitions)

    def sample(self, batch_size):
        
#         while len(self.batch_queue) < batch_size:
#             self.step_reverse_BFS()
            
        ## TODO: improve it by storing data in batches already so we don't need to pop many times
        batch_data = []
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for i in range(batch_size):
            
            state, action, reward, next_state, done, _ = self.batch_queue.pop()
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
#         print("sampled", np.array(states), np.array(actions), np.array(next_states))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
        
    def clear(self):
        self.buffer = dict()
        
    def prune_graph(self, last_t_step):
   
        transition = self.time_transitions[last_t_step]

        state = transition.state
        next_state = transition.next_state

        v = self.projection_matrix @ state
        vd = self.projection_matrix @ next_state

        v = tuple(v.T[0])
        vd = tuple(vd.T[0])

#         if self.verbose:
#             print(f'Removing transition from {v} to {vd} at {last_t_step}')
        self.buffer[vd][v].pop(last_t_step, 'None')
        if len(self.buffer[vd][v]) == 0:
            self.buffer[vd].pop(v, 'None')
            if len(self.buffer[vd]) == 0:
                self.buffer.pop(vd, 'None')
                if vd in self.terminal_vertices:
#                     print('Removing terminal vertex in prune --------------------------')
                    self.terminal_vertices.remove(vd)
                    if vd in self.search_queue:
                        self.search_queue.remove(vd)

        self.vertices = set(self.buffer.keys())
        for vd in self.buffer:
            self.vertices.update(set(self.buffer[vd].keys()))
                
        
#         for t in to_remove:
        del self.time_transitions[last_t_step]
        self.num_transitions -= 1
        
    def reset_terminal_vertices(self):
        initial_set = {}
        for vd in self.buffer:
            initial_set.add(vd)
           
        to_remove = {}
        for vd in self.buffer:
            to_remove.update(self.buffer[vd].keys())
#             for v in self.buffer[vd]:
#                 to_remove.add(v)
                
        self.terminal_vertices = initial_set - to_remove
        print('After Resetting:', self.terminal_vertices)
    
         
