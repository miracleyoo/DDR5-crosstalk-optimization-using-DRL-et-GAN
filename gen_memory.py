import numpy as np
import pickle

# NUM_TABLE = np.array([[[[50,50,75,100],[50,57,75,100],[100,100,100,100]],
# [[19,38,50,75],[50,50,57,75],[75,75,100,100]]],
# [[[25,38,50,100],[25,50,75,75],[100,100,100,100]],
# [[19,25,50,75],[25,50,75,100],[75,100,100,100]]]])

LEN_TABLE = np.array([1.5,2.0,3.0,4.0])
ACT_TABLE = np.array([2,0,1])
DR_TATBLE = np.arange(0.5,10.05,0.05)

COMPUTE_NUM_TABLE = False
if COMPUTE_NUM_TABLE:
    dataset = pickle.load(open("Datasets/direct_expanded_data_channel_comb_to10.pkl","rb"))
    val_range = pickle.load(open("source/val_range.pkl","rb"))
    NUM_TABLE = np.zeros([len(set([data[i] for data in dataset])) for i in range(4)])
    data_shape = NUM_TABLE.shape
    for i_idx, i in enumerate(sorted(set([data[0] for data in dataset]))):
        for j_idx, j in enumerate(sorted(set([data[1] for data in dataset]))):
            for k_idx, k in enumerate(sorted(set([data[2] for data in dataset]))):
                for l_idx, l in enumerate(sorted(set([data[3] for data in dataset]))):
                    rows = [row for row in dataset if row[0]==i and row[1]==j and row[2]==k and row[3]==l]
                    rows = sorted(rows,key=lambda x:x[-1])
                    # print(i_idx,j_idx,k_idx,l_idx)
                    if len(rows)>0:
                        NUM_TABLE[i_idx,j_idx,k_idx,l_idx] = rows[0][-2]
    pickle.dump(NUM_TABLE, open("./source/NUM_TABLE.pkl","wb+"))
else:
    NUM_TABLE=pickle.load(open("./source/NUM_TABLE.pkl","rb"))
    data_shape = NUM_TABLE.shape

memory = []

for i in range(data_shape[0]):
    for j in range(data_shape[1]):
        for k in range(data_shape[2]):
            for l in range(data_shape[3]):
                min_val = NUM_TABLE[i,j,k,l]
                if min_val != 0:
                    if min_val != 78:
                        ceil = 100
                    else:
                        ceil = 70
                    for val in range(ceil):
                        if abs(val-min_val)>=10:
                            sign = np.sign(val-min_val)
                            state = (i,j,np.around(DR_TATBLE[k],2),LEN_TABLE[l],val)
                            action = ACT_TABLE[int(sign)+1]
                            print(action, 1, state)
                            next_state = (i,j,np.around(DR_TATBLE[k],2),LEN_TABLE[l],int(val-sign))
                            memory.append((state, action, 1, next_state))
                            if sign != 0 and 0<=val+sign<=100:
                                action = ACT_TABLE[-int(sign)+1]
                                next_state = (i,j,np.around(DR_TATBLE[k],2),LEN_TABLE[l],int(val+sign))
                                memory.append((state, action, -1, next_state))

pickle.dump(memory, open("./source/generated_memory_to10.pkl","wb+"))