"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
import torch
import pickle
from gym import spaces, logger
from gym.utils import seeding
from icn_computing.utils import SPara
from gen_spara.para2icn import Generator
import numpy as np


class DDR5(gym.Env):
    def __init__(self):
        # length, tab_num
        self.low  = np.array([0, 0])
        self.high = np.array([1, 100])
        # (c1c2, spacing, dr, trace_len, tab_num)
        self.state_low = np.array([0,0,0,0,0])
        self.state_high = np.array([1,1,2,3,100])

        # Action: 0->No Action 1->-num_tab 2->+num_tab 3->change_c1c2
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.reward_range = (-2,2)

        self.seed()
        self.state = None
        self.steps_beyond_done = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.min_icn = 1
        self.min_icn_state = []

        self.frequencies = pickle.load(open('./source/frequencies.pkl','rb'))
        self.sparaNet = Generator().to(self.device)
        checkpoint = torch.load('./gen_spara/source/netG_direct_epoch_100.pth', map_location=self.device.type)
        self.sparaNet.load_state_dict(checkpoint)
        self.spara = SPara()
        self.norm_dict = [1,100]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_spara(self):
        parameters = []
        self.spara.Frequencies = np.array(self.frequencies)
        for freq in self.frequencies:
            inp = np.array([i/j for i,j in zip([*self.state, freq], self.norm_dict)])
            inp = inp[np.newaxis,:,np.newaxis,np.newaxis]
            inp = torch.from_numpy(inp).float().to(self.device)
            out = self.sparaNet(inp).detach().numpy().squeeze().transpose(1,2,0)
            parameters.append(out)
        parameters = np.array(parameters)
        self.spara.Parameters = np.zeros(
            (parameters.shape[0], parameters.shape[1], parameters.shape[1]), dtype=complex)
        for i in range(parameters.shape[0]):
            for j in range(parameters.shape[1]):
                for k in range(parameters.shape[1]):
                    self.spara.Parameters[i, j, k] = complex(
                        parameters[i, j, k, 0], parameters[i, j, k, 1])
        self.spara.NumPorts = self.spara.Parameters.shape[1]
        return self.spara

    def get_icn(self):
        ICNNet = Generator().to(self.device)
        checkpoint = torch.load('./gen_spara/source/netG_direct_epoch_100.pth', map_location=self.device.type)
        ICNNet.load_state_dict(checkpoint)
        norm_dict = [1,4000,1,100]
        inp = np.array([i/j for i,j in zip(self.state, norm_dict)])
        inp = inp[np.newaxis,:,np.newaxis,np.newaxis]
        inp = torch.from_numpy(inp).float().to(self.device)
        out = ICNNet(inp).detach().numpy()
        return out

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        c1c2, spacing, dr, trace_len, tab_num = state

        # computation
        if action == 0:
           pass
        elif action ==1:
            tab_num -= 1
            if tab_num<self.low[3]:
                tab_num +=2
        elif action ==2:
            tab_num += 1
            if tab_num>self.high[3]:
                tab_num -=2
        elif action ==3:
            c1c2 = int(not c1c2)
            
        self.state = (c1c2, spacing, dr, trace_len, tab_num)

        done =  False
        self.icn = self.get_icn()
        if self.icn<self.min_icn:
            self.min_icn = self.icn
            self.min_icn_state = state
        reward = min(max((self.last_icn - self.icn)*100,-2),2)
        self.last_icn = self.icn

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = [self.np_random.randint(self.state_low[i],self.state_high[i]) for i in range(len(self.state_high))]
        self.steps_beyond_done = None
        self.last_icn = self.get_icn()
        return np.array(self.state)
