# Author: Zhongyang Zhang
# E-mail: mirakuruyoo@gmail.com

import os
import pickle
from icn_computing.utils import *
from icn_computing.ICN_Main import sweep_files

# idx = 0: Constraint; idx = 1: Normal
roots = ['Datasets/meta/Normal_Tab_Constraint/','Datasets/meta/Normal_Tab_Normal_Spacing/']
tab_num_dict = [{'x01':0,'x101':10,'x201':20,'x301':30,'x401':40,'x501':50,'x601':60,
'x701':70,'x801':80,'x901':90,'x1001':100},
{'x101':0,'x201':10,'x301':20,'x401':30,'x501':40,'x601':50,
'x701':60,'x801':70,'x901':78}]
c1c2_dict = {'2010':0,'3020':1}
all_info_pack = []
SEP = False
CHANNEL = True
C1C2_COMB = True


def get_ICN(filename):
    # Configuration
    fb = [1.8*(10**9), 3*(10**9), 10*(10**9)][0]
    ft = fb
    fr = 0.75*fb  # the cut-off freq for the receiving filter [GHz]
    Ant = 1000*(10**-3)  # Disturber Amplitude @near end [v]
    Aft = 1000*(10**-3)  # Disturber Amplitude @far end [v]
    #  Port Order <from MATLAB s2smm()>
    #  1: pt1 and pt3 are pair 1 1->2 THU
    #  2: pt1 and pt2 are pair 1 1->N THU
    portOrder = 2
    sweepFileFigure = 0  # If plot figures while sweeping files: 0-NO 1-YES
    mode = 1  # SE or DIFF mode: 1-SingleEnded 2-Differential
    config = [Ant, Aft, fb, ft, fr, portOrder, sweepFileFigure, mode]
    prefix = './source/s-para/'

    #  Sweep all the s-parameter files
    temp = sweep_files(config, filename)
    return sum(temp[0])/len(temp[0])

for idx, root in enumerate(roots):
    paras = [i for i in os.listdir(root) if os.path.isdir(root+i) and not i.startswith('.')]
    paths = [root+i+'/' for i in paras]
    if C1C2_COMB:
        sep_paras = [[c1c2_dict[i.split('_')[0]],int(i.split('_')[1])] for i in paras]
    else:
        sep_paras = [[int(i.split('_')[0][:2]), int(i.split('_')[0][2:]),int(i.split('_')[1])] for i in paras]
    for path, para in zip(paths,sep_paras):
        names = [i for i in os.listdir(path) if i.endswith('.s8p')]
        tab_num = [tab_num_dict[idx][os.path.splitext(i)[0]] for i in names]
        sub_paths = [path+i for i in names]
        for afile, anum in zip(sub_paths,tab_num):
            icn = get_ICN(afile)
            info_pack = [*para, idx, anum, icn]
            all_info_pack.append(info_pack)

filename = 'Datasets/direct_data'
if CHANNEL:
    filename += '_channel'
if C1C2_COMB:
    filename += '_comb'
filename += '.pkl'
with open(filename,'wb+') as f:
    pickle.dump(all_info_pack,f)
    print("==> Pre-processing of "+ filename +' finished!')