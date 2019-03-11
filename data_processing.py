# Author: Zhongyang Zhang
# E-mail: mirakuruyoo@gmail.com

import os
import pickle
from icn_computing.utils import *

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
            spara = sparameters(afile)
            for j in range(len(spara.Frequencies)):
                if CHANNEL:
                    m=np.zeros((2,*spara.Parameters[j].shape))
                    for g in range(spara.Parameters[j].shape[0]):
                        for k in range(spara.Parameters[j].shape[1]):
                            m[0][g][k] = spara.Parameters[j][g,k].real
                            m[1][g][k] = spara.Parameters[j][g,k].imag
                    info_pack = [*para, idx, anum, int(round(spara.Frequencies[j])), m]
                else:
                    info_pack = [*para, idx, anum, int(round(spara.Frequencies[j])), spara.Parameters[j]]
                if SEP:
                    filename = 'Datasets/processed'
                    if CHANNEL:
                        filename += '_channel' 
                    if C1C2_COMB:
                        filename += '_comb'
                    filename += '/'+'_'.join([str(i) for i in info_pack[:-1]])+'.pkl'
                    with open(filename,'wb+') as f:
                        pickle.dump(info_pack,f)
                        print("==> Pre-processing of "+ filename +' finished!')
                else:
                    all_info_pack.append(info_pack)

if not SEP:
    filename = 'Datasets/all_data'
    if CHANNEL:
        filename += '_channel'
    if C1C2_COMB:
        filename += '_comb'
    filename += '.pkl'
    with open(filename,'wb+') as f:
        pickle.dump(all_info_pack,f)
        print("==> Pre-processing of "+ filename +' finished!')