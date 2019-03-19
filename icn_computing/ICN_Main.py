# Author: Zhongyang Zhang
# E-mail: mirakuruyoo@gmail.com

import os
import pickle
from .utils import *


def sweep_files(config, filename=None, obj0=None):
    if filename is not None:
        print("Now processing: ", filename)
        obj0 = sparameters(filename)
    elif obj0 is not None:
        # print("==> Using input s-parameters.")
        filename = 'temp.s8p'
    else:
        raise KeyError("==> Please at least input a filename or a s-parameter object!")
    s_freq = obj0.Frequencies
    s_freq = np.transpose(s_freq)
    portOrder = config[5]
    sweepFileFigure = config[6]
    ct_nx = []
    ct_fx = []
    ct_total = []

    ##### Sweep all the victims: Port order 1 or 2 #####
    ### Get the port number ###
    s0 = obj0.Parameters
    portNum = min(s0.shape)

    if portOrder == 2:  # 1->N THU
        victimSEQ = list(range(int(0.5*portNum)))
    elif portOrder == 1:  # 1->2 THU
        victimSEQ = list(range(0, portNum, 2))

    #### Get ICN for all the victims ###
    for v in range(len(victimSEQ)):
        temp = ICN(obj0, victimSEQ[v], config)
        ct_nx.append(temp[0])
        ct_fx.append(temp[1])
        ct_total.append(temp[2])

    ct_nx = np.array(ct_nx)
    ct_fx = np.array(ct_fx)
    ct_total = np.array(ct_total)

    #####  OUTPUT: TXT files &. Figures  ######
    sampleNum = ct_nx.shape[0]
    ####### Final Value ###########
    final_ct_nx = []
    final_ct_fx = []
    final_ct_total = []
    for d in range(sampleNum):
        final_ct_nx.append(ct_nx[d, -1])
        final_ct_fx.append(ct_fx[d, -1])
        final_ct_total.append(ct_total[d, -1])

    ########### TXT output #########
    if not os.path.exists('./source/cache/'): os.mkdir('./source/cache/')
    output_filename = './source/cache/' + \
        os.path.splitext((os.path.split(filename)[1]))[0]+'.txt'
    with open(output_filename, 'w+') as f:
        f.write("NUM\tFA\tNE\tTT\n")
        for i in range(sampleNum):
            f.write(str(i)+'\t'+str(final_ct_fx[i])+'\t' +
                    str(final_ct_nx[i])+'\t'+str(final_ct_total[i])+'\n')

    return final_ct_fx, final_ct_nx, final_ct_total


def ICN(obj, victim, config):
    f = obj.Frequencies
    s0 = obj.Parameters

    Ant = config[0]  # Disturber Amplitude @near  [mv]
    Aft = config[1]  # Disturber Amplitude @far  [mv]
    fb = config[2]  # data rate 30[G bps]
    ft = config[3]
    fr = config[4]  # the cut-off freq for the receiving filter [GHz]
    ptOrder = config[5]
    mode = config[7]

    ###################################
    ########   Select Modes   #########
    ###################################
    ##### Single ed Mode: mode=1  #####
    if mode == 1:
        sdd = s0
    #### Differential Mode: mode=2  ####
    elif mode == 2:
        sdd, sdc, scd, scc = s2smm(s0, ptOrder)

    ############################
    ### Weight function   ######
    ############################

    weightNT = []
    weightFT = []
    for i in range(len(f)):
        weightNT.append(
            ((Ant**2)/fb) * ((np.sinc(f[i]/fb))**2) * (1/(1 + (f[i]/ft)**4)) * (1/(1+(f[i]/fr)**8)))
        weightFT.append(
            ((Aft**2)/fb) * ((np.sinc(f[i]/fb))**2) * (1/(1 + (f[i]/ft)**4)) * (1/(1+(f[i]/fr)**8)))

    ########################################
    ### Multi-disturber crosstalk  loss ####
    ########################################
    [MDNEXT, MDFEXT] = MDXT(sdd, victim, ptOrder)
    deltaF = f[2]-f[1]
    sum_nx = 0  # Initial Value of sum
    sum_fx = 0
    ct_nx = []
    ct_fx = []
    for j in range(len(f)):
        sum_nx = sum_nx + weightNT[j]*10**(-0.1*MDNEXT[j])
        ct_nx.append((2*deltaF * sum_nx)**(0.5))  # Near- Crosstalk

        sum_fx = sum_fx + weightFT[j]*10**(-0.1*MDFEXT[j])
        ct_fx.append((2*deltaF * sum_fx)**(0.5))  # Far- Crosstalk

    ct_nx = np.array(ct_nx)
    ct_fx = np.array(ct_fx)
    #### Totla ICN ######
    ct_total = (ct_nx**2 + ct_fx**2)**(0.5)
    return ct_nx, ct_fx, ct_total

def get_ICN(obj0, fb=1.8*(10**9)):
    # Configuration
    ft = fb
    fr = 0.75*fb  # the cut-off freq for the receiving filter [GHz]
    Ant = 1000*(10**-3)  # Disturber Amplitude @near end [v]
    Aft = 1000*(10**-3)  # Disturber Amplitude @far end [v]
    portOrder = 2
    sweepFileFigure = 0  # If plot figures while sweeping files: 0-NO 1-YES
    mode = 1  # SE or DIFF mode: 1-SingleEnded 2-Differential
    config = [Ant, Aft, fb, ft, fr, portOrder, sweepFileFigure, mode]

    #  Sweep all the s-parameter files
    temp = sweep_files(config, obj0=obj0)
    FA = temp[0]
    return sum(FA)/(len(FA))

def main():
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
    filenames = [prefix+i for i in os.listdir(prefix) if i.endswith('.s8p')]

    #  Sweep all the s-parameter files
    FA = []
    NE = []
    TT = []
    for filename in filenames:
        temp = sweep_files(config, filename)
        FA.append(temp[0])
        NE.append(temp[1])
        TT.append(temp[2])
    pickle.dump([FA, NE, TT], open('./source/cache/result.pkl', 'wb'))


if __name__ == "__main__":
    main()
