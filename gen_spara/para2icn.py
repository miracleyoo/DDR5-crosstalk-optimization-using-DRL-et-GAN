# Author: Zhongyang Zhang
# E-mail: mirakuruyoo@gmail.com

'''
Models used in the generating network.
'''

import torch.nn as nn

class Generator0(nn.Module):
    def __init__(self, ngpu=1, nz=3, ngf=64, nc=2):
        super(Generator0, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nz, ngf*4),
            nn.BatchNorm1d(ngf*4),
            nn.ReLU(),
            nn.Linear(ngf*4, ngf*2),
            nn.BatchNorm1d(ngf*2),
            nn.ReLU(),
            nn.Linear(ngf*2, ngf),
            nn.BatchNorm1d(ngf),
            nn.ReLU(),
            nn.Linear(ngf, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.nz)
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output.squeeze()


class Generator0NS(nn.Module):
    def __init__(self, ngpu=1, nz=3, ngf=64, nc=2):
        super(Generator0NS, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nz, ngf*4),
            nn.BatchNorm1d(ngf*4),
            nn.ReLU(),
            nn.Linear(ngf*4, ngf*2),
            nn.BatchNorm1d(ngf*2),
            nn.ReLU(),
            nn.Linear(ngf*2, ngf),
            nn.BatchNorm1d(ngf),
            nn.ReLU(),
            nn.Linear(ngf, 1),
        )

    def forward(self, x):
        x = x.view(-1, self.nz)
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output.squeeze()


class Generator1(nn.Module):
    def __init__(self, ngpu=1, nz=3, ngf=64, nc=2):
        super(Generator1, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nz, ngf*4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(ngf*4),
            nn.Linear(ngf*4, ngf*2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(ngf*2),
            nn.Linear(ngf*2, ngf),
            nn.Tanh(),
            nn.BatchNorm1d(ngf),
            nn.Linear(ngf, 1)
        )

    def forward(self, x):
        x = x.view(-1, self.nz)
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output.squeeze()


class Generator2(nn.Module):
    def __init__(self, ngpu=1, nz=3, ngf=64, nc=2):
        super(Generator2, self).__init__()
        self.nc = nc
        self.ngpu = ngpu
        self.main1 = nn.Sequential(
            nn.ConvTranspose2d(     nz, ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ngf x 4 x 4
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(True)
        )
            # state size. nc x 8 x 8
        self.main2 = nn.Sequential(
            nn.Linear(nc*8*8, ngf),
            nn.BatchNorm1d(ngf),
            nn.ReLU(),
            nn.Linear(ngf, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main1, x, range(self.ngpu))
            output = output.view(-1, self.nc*8*8)
            output = nn.parallel.data_parallel(self.main2, output, range(self.ngpu))
        else:
            output = self.main1(x)
            output = output.view(-1, self.nc*8*8)
            # print(output.size())
            output = self.main2(output)
        return output.squeeze()




# class SParaData(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, opt):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         with open(opt.dataroot,'rb') as f:
#             self.dataset = pickle.load(f)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         norm_dict = [1,4000,1,100,11910000000]
#         # norm_dict = [1,1,1,1,1]
        
#         inputs = np.array([i/j for i,j in zip(self.dataset[idx][:-1], norm_dict)])
#         inputs = inputs[:,np.newaxis,np.newaxis]
#         labels = self.dataset[idx][-1]
#         return torch.from_numpy(inputs).float(), torch.FloatTensor(labels)

# dataset = SParaData(opt)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                          shuffle=True, num_workers=int(opt.workers))



# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square you can only specify a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features