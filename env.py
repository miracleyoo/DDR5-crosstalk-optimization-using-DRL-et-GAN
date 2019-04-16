# Author: Zhongyang Zhang
# E-mail: mirakuruyoo@gmail.com

'''
The main environment of the project. 
'''

import math
import gym
import torch
import pickle
from gym import spaces, logger
from gym.utils import seeding
from icn_computing.utils import SPara
from gen_spara.para2icn import *
import numpy as np


class DDR5(gym.Env):
    def __init__(self):
        with open('./source/val_range.pkl', 'rb') as f:
            self.val_range = pickle.load(f)
        self.norm_dict = self.val_range["high"]
        # (c1c2, spacing, dr, trace_len, tab_num)
        self.low = np.array(self.val_range["low"])
        self.high = np.array(self.val_range["high"])

        # Action: 0->No Action 1->-num_tab 2->+num_tab 3->-length 4->+length 5->change_c1c2
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32)
        self.reward_range = (-2, 2)

        self.seed()
        self.state = None
        self.steps_beyond_done = None
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.min_icn = 1
        self.min_icn_state = []

        self.frequencies = pickle.load(open('./source/frequencies.pkl', 'rb'))
        self.sparaNet = [[], []]
        PREFIX = "gen_spara/source/G0/G0_gened_data_sep_L1_NEW_TO10_IMI/"
        net_paths = [["netG0_direct_choice_0_0.pth", "netG0_direct_choice_0_1.pth"],
                     ["netG0_direct_choice_1_0.pth", "netG0_direct_choice_1_1.pth"]]
        for idx_1 in range(2):
            for idx_2 in range(2):
                temp_net = Generator0(nz=3).to(self.device)
                temp_net.load_state_dict(torch.load(
                    PREFIX+net_paths[idx_1][idx_2], map_location=self.device.type))
                temp_net.eval()
                self.sparaNet[idx_1].append(temp_net)
                del temp_net
        self.spara = SPara()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_spara(self):
        parameters = []
        self.spara.Frequencies = np.array(self.frequencies)
        for freq in self.frequencies:
            inp = np.array(
                [i/j for i, j in zip([*self.state, freq], self.norm_dict)])
            inp = inp[np.newaxis, :, np.newaxis, np.newaxis]
            inp = torch.from_numpy(inp).float().to(self.device)
            out = self.sparaNet(inp).detach(
            ).numpy().squeeze().transpose(1, 2, 0)
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
        raw = np.array([i/j for i, j in zip(self.state, self.norm_dict)])
        inp = raw[-3:][np.newaxis, :, np.newaxis, np.newaxis]
        inp = torch.from_numpy(inp).float().to(self.device)
        # print("inp:",inp,self.norm_dict,raw)
        # out = ICNNet(inp).detach().numpy()
        out = self.sparaNet[int(raw[0])][int(raw[1])](inp).detach().numpy()
        return out

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        spacing, c1c2, dr, trace_len, tab_num = self.state

        # computation
        if action == 0:
           pass
        elif action == 1:
            tab_num -= 1
            if tab_num < self.low[-1]:
                tab_num += 2
        elif action == 2:
            tab_num += 1
            if tab_num > self.high[-1]:
                tab_num -= 2
        elif action == 3:
            trace_len -= 0.1
            if np.around(trace_len, 2) < self.low[-2]:
                trace_len += 0.2
        elif action == 4:
            trace_len += 0.1
            if np.around(trace_len, 2) > self.high[-2]:
                trace_len -= 0.2
        elif action == 5:
            c1c2 = int(not c1c2)

        self.state = (spacing, c1c2, dr, np.around(trace_len, 2), tab_num)

        done = False
        self.icn = self.get_icn()
        if self.icn < self.min_icn:
            self.min_icn = self.icn
            self.min_icn_state = self.state
        # -0.1*(self.last_icn == self.icn)
        reward = min(max((self.last_icn - self.icn)*10e2, -1), 1) - \
            0.05*(self.last_icn == self.icn)
        self.last_icn = self.icn

        return np.array(self.state), reward, done, {}

    def reset(self, init_state=None):
        # self.state = [self.np_random.randint(self.low[i],self.high[i]) for i in range(len(self.high))]
        if init_state is None:
            self.state = [0, 0, np.around(0.1*self.np_random.randint(10*self.low[2], 10*self.high[2])),
                          np.around(0.1*self.np_random.randint(10 * self.low[3], 10*self.high[3]), 2),
                          self.np_random.randint(self.low[4], self.high[4])]
        else:
            try:
                self.state = init_state
            except (AttributeError, TypeError):
                raise AssertionError(
                    'Input init_state should be a state array!')
        self.steps_beyond_done = None
        self.last_icn = self.get_icn()
        return np.array(self.state)
