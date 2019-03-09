# Author: Zhongyang Zhang
# E-mail: mirakuruyoo@gmail.com

from math import log10
import numpy as np


class SPara(object):
    def __init__(self):
        self.Frequencies = np.array([])
        self.NumPorts = 0
        self.Parameters = np.array([[]])
        self.Impedance = 50


def sparameters(filename):
    with open(filename, 'r') as f:
        q = f.read()
    sp = [i for i in q.splitlines() if i != '' and i[0] not in ['!', '#']]
    sp_num = [[float(i) for i in p.split(' ') if i != ''] for p in sp]
    spara = SPara()

    parameters = []
    frequencies = []
    temp = []
    continue_flag = True
    for data in sp_num:
        if len(data) == 9:
            if temp != []:
                parameters.append(np.array(temp))
            temp = []
            if data[0] < 100:
                frequencies.append(data[0]*(10**9))
            else:
                frequencies.append(data[0])
            temp.append(data[1:])
            continue_flag = True
        else:
            if continue_flag:
                temp[-1].extend(data)
                continue_flag = False
            else:
                temp.append(data)
                continue_flag = True
    parameters.append(np.array(temp))

    parameters = np.array(parameters)
    spara.Parameters = np.zeros(
        (parameters.shape[0], parameters.shape[1], parameters.shape[1]), dtype=complex)
    for i in range(parameters.shape[0]):
        for j in range(parameters.shape[1]):
            for k in range(parameters.shape[1]):
                spara.Parameters[i, j, k] = complex(
                    parameters[i, j, 2*k], parameters[i, j, 2*k+1])
    spara.Frequencies = np.array(frequencies)
    spara.NumPorts = spara.Parameters.shape[1]
    return spara


def MDXT(sdd, vic, ptOrder):
    #######################################
    #### aggressors identification ########
    #######################################
    portNum = min(sdd.shape)
    NE_agg = []
    FE_agg = []
    #####  THU: 1->n+1  ######
    if ptOrder == 2:
        for i in range(int(0.5*portNum)):
            if vic != i:
                NE_agg.append(i)
                FE_agg.append(int(i+(0.5*portNum)))
    ######  THU: 1->2  ######
    elif ptOrder == 1:
        for i in range(int(0.5*portNum)):
            if vic != (2*i+1):
                NE_agg.append(2*i)
                FE_agg.append(2*i+1)

    print('NE_AGG:', NE_agg, '\nFE_AGG:', FE_agg, ' ')

    ###################################
    #### Get all the NE and FE ########
    ###################################
    NE = np.zeros((len(NE_agg), len(sdd)), dtype=complex)
    FE = np.zeros((len(NE_agg), len(sdd)), dtype=complex)

    for kk in range(len(NE_agg)):
        for ii in range(len(sdd)):
            # ALL THE FAR-END
            NE[kk, ii] = sdd[ii, NE_agg[kk], vic]
            # ALL THE FAR-END
            FE[kk, ii] = sdd[ii, FE_agg[kk], vic]

    #########################################
    #### Get the accunulative NE&FE  ########
    #########################################
    SUM_NE = 0
    SUM_FE = 0

    for jj in range(len(NE_agg)):
        SUM_NE += np.abs(NE[jj, :])**2
        SUM_FE += np.abs(FE[jj, :])**2

    MDNEXT = np.array([-10 * log10(k) for k in SUM_NE])
    MDFEXT = np.array([-10 * log10(k) for k in SUM_FE])

    return MDNEXT, MDFEXT


def s2smm(s_params, rfflag):

    # Determine the size of the (3-D) S-parameter matrix
    N = s_params.shape[2]
    pts = s_params.shape[0]

    # Determine the mixed-mode matrices according to the output.
    # If the number of ports is even, calculate differential & common modes
    Ports = list(range(0, N-1, 2))
    Ports.extend(list(range(1, N, 2)))
    # Check the specified ordering for S-parameters and reorder if necessary
    if rfflag == 1:
        Ports = list(range(0, N-1, 2))
        Ports.extend(list(range(1, N, 2)))
    elif rfflag == 2:
        Ports = range(N)
    elif rfflag == 3:
        Ports = list(range(0, N//2))
        Ports.extend(list(range(N-1, N//2-1, -1)))
    else:
        raise AttributeError("rfflag wrong!")

    # s_params = s_params(Ports,Ports,:)
    s_params = s_params[:, Ports, :][:, :, Ports]

    # Create the transformation matrix
    M1 = []
    M2 = []
    for idx in range(N//2):
        M1 = np.eye(len(M1)+2)*(np.hstack(M1, [1, -1]))
        M1 = np.eye(len(M2)+2)*(np.hstack(M2, [1, 1]))
    M = np.vstack(M1, M2)
    invM = M.T

    # Apply the transformation matrix
    smm_params = np.zeros(pts, N, N)
    for idx in range(pts):
        smm_params[idx, :, :] = M*s_params[idx, :, :]*invM/2

    varargout = []
    varargout.append(smm_params[:, 0:N//2, :][:, :, 0:N//2])
    varargout.append(smm_params[:, N//2:N, :][:, :, 0:N//2])
    varargout.append(smm_params[:, 0:N//2, :][:, :, N//2:N])
    varargout.append(smm_params[:, N//2:N, :][:, :, N//2:N])

    return varargout
