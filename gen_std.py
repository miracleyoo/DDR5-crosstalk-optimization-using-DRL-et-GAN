# Author: Zhongyang Zhang
# E-mail: mirakuruyoo@gmail.com

'''
Generate dataset from the ground truth network and pack them into a pickle file.
'''

import argparse
import os
import time
import random
import torch
import pickle

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gen_spara.para2icn import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='../Datasets/matlab_direct_expanded_data_channel_comb_to10.pkl', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--nz', type=int, default=3, help='size of the input vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--outf', default='./source/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

def log(*args, end=None):
    if end is None:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in args]))
    else:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in args]),
              end=end)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


class SParaData(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, opt, choice):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = []
        for i in range(2):
            for j in range(2):
                for k in np.arange(0.5,10.05,0.05):
                    for l in np.arange(1.5,4.05,0.05):
                        for p in np.arange(0,101,2):
                            self.dataset.append([i,j,k,l,p])
        with open('./source/val_range.pkl','rb') as f:
            self.val_range = pickle.load(f)
        self.dataset = [i for i in self.dataset if i[0]==choice[0] and i[1]==choice[1]]
   
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        norm_dict = self.val_range["high"][2:]
        inputs = np.array([i/j for i,j in zip(self.dataset[idx][2:], norm_dict)])
        inputs = inputs[:,np.newaxis,np.newaxis]
        return torch.from_numpy(inputs).float()

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
nc = 2
criterion = nn.L1Loss()
DECAY_RATE = 0.9
lr = opt.lr

PREFIX = "./gen_spara/source/G0/G0_matlab_sep_L1_New_TO10/"
net_paths = [["netG0_direct_choice_0_0.pth","netG0_direct_choice_0_1.pth"],
["netG0_direct_choice_1_0.pth","netG0_direct_choice_1_1.pth"]]

log("Start test!")
gened_dataset = []

for idx_1 in range(2):
    for idx_2 in range(2):
        min_loss = 1
        dataset = SParaData(opt, (idx_1, idx_2))
        norm_dict = dataset.val_range["high"][2:]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                                shuffle=False, num_workers=0)

        netG = Generator0(ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
        checkpoint = torch.load(PREFIX+net_paths[idx_1][idx_2], map_location=device.type)
        netG.load_state_dict(checkpoint)
        netG.eval()

        for i, data in enumerate(dataloader):
            inputs = data
            inputs = inputs.to(device)

            outputs = netG(inputs)
            gened_dataset.extend([[idx_1, idx_2, *list(np.around([k*l for k,l in zip(i, norm_dict)],2)), j] for i ,j in zip(inputs.squeeze().tolist(), outputs.squeeze().tolist())])
pickle.dump(gened_dataset, open("./source/generated_dataset.pkl","wb"))