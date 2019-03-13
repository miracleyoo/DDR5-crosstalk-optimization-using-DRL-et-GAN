import torch.nn as nn
import torch.nn.parallel

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=5, ngf=64, nc=2):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ngf x 4 x 4
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False)
            # nn.Tanh()
            # state size. nc x 8 x 8
        )
        # self.main = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(     nz, ngf*8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf*8),
        #     nn.ReLU(True),
        #     # state size. ngf x 4 x 4
        #     nn.ConvTranspose2d(    ngf*8,      ngf*4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf*4),
        #     nn.ReLU(True),
        #     # state size. nc x 8 x 8
        #     nn.ConvTranspose2d(    ngf*4,      ngf*2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf*2),
        #     nn.ReLU(True),
        #     # state size. nc x 16 x 16
        #     nn.ConvTranspose2d(    ngf*2,      ngf*2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf*2),
        #     nn.ReLU(True),
        #     # state size. nc x 32 x 32       
        #     nn.Conv2d(    ngf*2,      ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. nc x 16 x 16
        #     nn.Conv2d(    ngf,      nc, 4, 2, 1, bias=False)
        #     # state size. nc x 8 x 8                          
        # )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

