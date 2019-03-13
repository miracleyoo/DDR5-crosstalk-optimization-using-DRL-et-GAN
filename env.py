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
from gen_spara.module import Generator
import numpy as np


class DDR5(gym.Env):
    def __init__(self):
        # length, tab_num
        self.low  = np.array([0, 1500, 0, 0])
        self.high = np.array([1, 4000, 1, 100])

        # Action: 0->No Action 1->-length 2->+length 3->-num_tab 4->+num_tab 5->change_c1c2 6->change_spacing
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.reward_range = (-2,2)

        self.seed()
        self.state = None
        self.steps_beyond_done = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_spara(self):
        frequencies = pickle.load(open('./source/frequencies.pkl','rb'))
        sparaNet = Generator().to(self.device)
        checkpoint = torch.load('./gen_spara/source/netG_epoch_10.pth', map_location=self.device.type)
        sparaNet.load_state_dict(checkpoint)
        spara = SPara()
        norm_dict = [1,4000,1,100,11910000000]
        parameters = []
        spara.Frequencies = np.array(frequencies)
        for freq in frequencies:
            inp = np.array([i/j for i,j in zip([*self.state, freq], norm_dict)])
            inp = inp[np.newaxis,:,np.newaxis,np.newaxis]
            inp = torch.from_numpy(inp).float().to(self.device)
            out = sparaNet(inp).detach().numpy().squeeze().transpose(1,2,0)/10e5
            parameters.append(out)
        parameters = np.array(parameters)
        spara.Parameters = np.zeros(
            (parameters.shape[0], parameters.shape[1], parameters.shape[1]), dtype=complex)
        for i in range(parameters.shape[0]):
            for j in range(parameters.shape[1]):
                for k in range(parameters.shape[1]):
                    spara.Parameters[i, j, k] = complex(
                        parameters[i, j, k, 0], parameters[i, j, k, 1])
        spara.NumPorts = spara.Parameters.shape[1]
        return spara

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        c1c2, trace_len, spacing, tab_num = state

        # computation
        if action == 0:
           pass
        if action == 1:
            trace_len -= 1
        elif action == 2:
            trace_len += 1
        elif action ==3:
            tab_num -= 1
        elif action ==4:
            tab_num += 1
        elif action ==5:
            self.c1c2 = not self.c1c2
        elif action ==6:
            self.spacing = not self.spacing
            
        self.state = (c1c2, trace_len, spacing, tab_num)

        done =  False
        icn = get_ICN(obj0=self.get_spara())
        reward = min(max((self.last_icn - icn)*100,-2),2)
        self.last_icn = icn

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = [self.np_random.randint(self.low[i],self.high[i]) for i in range(len(self.low))]
        self.steps_beyond_done = None
        self.last_icn = get_ICN(obj0=self.get_spara())
        return np.array(self.state)
