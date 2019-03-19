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
from icn_computing.ICN_Main import get_ICN
from icn_computing.utils import SPara
from gen_spara.para2icn import Generator
import numpy as np


class DDR5(gym.Env):
    def __init__(self):
        # length, tab_num
        self.low  = np.array([0, 1500, 0, 10])
        self.high = np.array([1, 4000, 1, 100])

        # Action: 0->No Action 1->-length 2->+length 3->-num_tab 4->+num_tab 5->change_c1c2 6->change_spacing
        self.action_space = spaces.Discrete(7)
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
        self.norm_dict = [1,4000,1,100,11910000000]

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
        # print(out)
        return out

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        c1c2, trace_len, spacing, tab_num = state

        # computation
        if action == 0:
           pass
        if action == 1:
            trace_len -= 10
            if trace_len<self.low[1]:
                trace_len +=20
        elif action == 2:
            trace_len += 10
            if trace_len>self.high[1]:
                trace_len -=20
        elif action ==3:
            tab_num -= 1
            if tab_num<self.low[3]:
                tab_num +=2
        elif action ==4:
            tab_num += 1
            if tab_num>self.high[3]:
                tab_num -=2
        elif action ==5:
            c1c2 = int(not c1c2)
        elif action ==6:
            spacing = int(not spacing)
            
        self.state = (c1c2, trace_len, spacing, tab_num)

        done =  False
        icn = self.get_icn()#get_ICN(obj0=self.get_spara())
        if icn<self.min_icn:
            self.min_icn = icn
            self.min_icn_state = state
        reward = min(max((self.last_icn - icn)*100,-2),2)
        self.last_icn = icn

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = [self.np_random.randint(self.low[i],self.high[i]) for i in range(len(self.low))]
        self.steps_beyond_done = None
        self.last_icn = self.get_icn()#get_ICN(obj0=self.get_spara())
        return np.array(self.state)
