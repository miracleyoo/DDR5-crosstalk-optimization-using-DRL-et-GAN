# Author: Zhongyang Zhang
# E-mail: mirakuruyoo@gmail.com

'''
DQN main file. Can execute test and train process.
'''

import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
import copy
import random
from tqdm import tqdm
from time import time
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
from env import DDR5
import pandas as pd
import numpy as np                                                                                                                            
import seaborn as sns                                             
import matplotlib.pyplot as plt
sns.set()
sns.set_context("paper", font_scale=1.4)


# hyper-parameters
INIT_DQN = False
BATCH_SIZE = 1024
LR = 0.0001
GAMMA = 0.90
EPISILO = 0.9
EPISODES = 2000
MEMORY_CAPACITY = 4096
Q_NETWORK_ITERATION = 100
MAX_STEP = 256
INIT_TRAIN_EPOCH = 5
INNER_NUM = 256
OUT_IMG_PATH = './source/result/reward_graph.jpg'

env = DDR5()
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(
    env.action_space.sample(), int) else env.action_space.sample.shape



class Net(nn.Module):
    """docstring for Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(NUM_STATES, INNER_NUM*4),
            nn.LeakyReLU(),
            nn.Linear(INNER_NUM*4, INNER_NUM),
            nn.LeakyReLU(),
            nn.Linear(INNER_NUM, NUM_ACTIONS)
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out


class InitData(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open('./source/generated_memory_to10.pkl', 'rb') as f:
            self.init_data = pickle.load(f)
        self.init_data = [i for i in self.init_data if i[0][0] ==
                          0 and i[0][1] == 0 and i[0][2] == 1.8 and i[0][3] == 1.5]

    def __len__(self):
        return len(self.init_data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.init_data[idx][0]), torch.LongTensor([self.init_data[idx][1]]), torch.FloatTensor([self.init_data[idx][2]]), torch.FloatTensor(self.init_data[idx][3])


class DQN():
    """docstring for DQN"""

    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net.eval()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array
        if np.random.rand() <= EPISILO:  # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        else:  # random policy
            action = np.random.randint(0, NUM_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Update the parameters
        self.eval_net.train()
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(
            batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(
            batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        # Q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.eval_net.eval()

    def init_learn(self):
        # Sample batch from memory
        init_dataset = InitData()
        dataloader = torch.utils.data.DataLoader(init_dataset, batch_size=BATCH_SIZE,
                                                 shuffle=True, num_workers=0, drop_last=False)
        for epoch in range(INIT_TRAIN_EPOCH):
            for i, data in tqdm(enumerate(dataloader), desc="Training", total=len(dataloader), leave=False, unit='b'):
                batch_state, batch_action, batch_reward, batch_next_state = data
                # Q_eval
                q_eval = self.eval_net(batch_state).gather(1, batch_action)
                q_next = self.target_net(batch_next_state).detach()
                q_target = batch_reward + GAMMA * \
                    q_next.max(1)[0].view(len(batch_reward), 1)
                loss = self.loss_func(q_eval, q_target)
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print("Epoch {} Finished training!".format(epoch))
            # Update the parameters of Target Net
            if epoch % 5 == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
        self.eval_net.eval()


def test():
    ###################################################
    # Initial Parameters Here
    ###################################################
    # state = [0, 0, 1.8, 3, 0]
    GATE_VAL = 0.15
    TEST_EPOCH = 100
    MAX_ROUND = 500
    logs = []
    dqn = DQN()
    dqn.eval_net.load_state_dict(torch.load(
        './source/dqn_1.pth', map_location=device.type))
    for epoch in range(TEST_EPOCH):
        state = env.reset()
        state_start = deepcopy(state)
        counter = 0
        tstart = time()
        while counter < MAX_ROUND:
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            counter += 1
            print(next_state, reward, env.icn)
            if done:
                break
        time_ela = time() - tstart
        logs.append([time_ela, env.icn, env.min_icn, counter, state_start])
        print("Time elapsed:{}".format(time_ela))
        print("min_icn:", env.min_icn, "min_icn_state:", env.min_icn_state)
    times = [i[0] for i in logs]
    print("Avg time:{}".format(sum(times)/len(times)))
    pickle.dump(logs, open("./source/dqn_test_log.pkl",'wb'))


def main():
    dqn = DQN()
    print("Pretraining on artificial memory...")
    # if INIT_DQN:
    #     dqn.init_learn()
    #     torch.save(dqn.eval_net.state_dict(), './source/dqn_init_epoch_5.pth')
    # else:
    #     dqn.eval_net.load_state_dict(
    #         torch.load('./source/dqn_init_epoch_5.pth'))
    # dqn.target_net.load_state_dict(dqn.eval_net.state_dict())
    print("Collecting Experience....")
    reward_list = []
    logs = []
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(EPISODES):
        state = env.reset()
        ep_reward = 0
        counter = 0
        while counter < MAX_STEP:
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)
            print(next_state, reward, env.icn)
            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward
            if done:
                print("episode: {} , the episode reward is {}".format(
                        i, round(ep_reward, 3)))
                break
            state = next_state
            counter += 1
        if dqn.memory_counter >= MEMORY_CAPACITY:
            dqn.learn()
        print("min_icn:", env.min_icn, "min_icn_state:", env.min_icn_state)
        if counter==1:
            i -= 1
            continue
        else:
            reward_list.append(ep_reward/(counter+1))#(r+reward_list[-1])
            logs.append((state, action, reward, next_state, env.icn, env.min_icn, counter, reward_list[-1]))
        ax.clear()
        ax.set_xlim(0, EPISODES)
        sns.lineplot(x='x',y='y',data=pd.DataFrame({'x':list(range(len(reward_list))),'y':reward_list}),ax=ax)
        # ax.plot(reward_list, '-', color='steelblue', label='total_loss')
        plt.pause(0.001)
    plt.savefig(OUT_IMG_PATH, format='png', dpi=1000)
    torch.save(dqn.eval_net.state_dict(), './source/dqn_1.pth')
    pickle.dump(logs, open("./source/dqn_train_log.pkl",'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='IF Test')
    opt = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(opt)
    if opt.test:
        test()
    else:
        main()
