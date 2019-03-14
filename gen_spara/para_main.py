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
from module import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='../Datasets/all_data_channel_comb.pkl', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=5, help='size of the input vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--outf', default='./source/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

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

    def __init__(self, opt):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(opt.dataroot,'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        norm_dict = [1,4000,1,100,11910000000]
        # norm_dict = [1,1,1,1,1]
        
        inputs = np.array([i/j for i,j in zip(self.dataset[idx][:-1], norm_dict)])
        inputs = inputs[:,np.newaxis,np.newaxis]
        labels = self.dataset[idx][-1]
        return torch.from_numpy(inputs).float(), torch.from_numpy(labels).float()

dataset = SParaData(opt)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
nc = 2

def log(*args, end=None):
    if end is None:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in args]))
    else:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in args]),
              end=end)
            
netG = Generator(ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
# netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

criterion = nn.MSELoss()

# setup optimizer

log("Start training!")
DECAY_RATE = 0.5
lr = opt.lr

for epoch in range(opt.niter):
    optimizer = optim.Adam(netG.parameters(), lr=lr)
    for i, data in tqdm(enumerate(dataloader), desc="Training", total=len(dataloader), leave=False,
                            unit='b'):
        inputs, labels = data
        labels = labels * 10e5
        inputs, labels = inputs.to(device), labels.to(device)
        print('inputs:',inputs[0],'labels:',labels[0])

        outputs = netG(inputs)

        loss = criterion(outputs, labels)
        print('outputs:',outputs[0],'loss:',loss)
        loss.backward()

        optimizer.step()
    lr = max(lr * DECAY_RATE, 0.0001)
    # print log
    log('Epoch [%d/%d], Train Loss: %.4f' % (epoch + 1, opt.niter, loss))

    # do checkpointing
torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch+1))