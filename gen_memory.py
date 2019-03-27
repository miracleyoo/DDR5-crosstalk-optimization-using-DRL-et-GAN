import numpy as np
import pickle

# NUM_TABLE = np.array([[[[50,50,75,100],[50,57,75,100],[100,100,100,100]],
# [[19,38,50,75],[50,50,57,75],[75,75,100,100]]],
# [[[25,38,50,100],[25,50,75,75],[100,100,100,100]],
# [[19,25,50,75],[25,50,75,100],[75,100,100,100]]]])

LEN_TABLE = np.array([1500,2000,3000,4000])
ACT_TABLE = np.array([2,0,1])

dataset = pickle.load(open("Datasets/direct_expanded_data_channel_comb_to10.pkl","rb"))
val_range = pickle.load(open("source/val_range.pkl","rb"))
NUM_TABLE = np.zeros([len(set([data[i] for data in dataset])) for i in range(4)])
data_shape = NUM_TABLE.shape
for i_idx, i in enumerate(set([data[0] for data in dataset])):
    for j_idx, j in enumerate(set([data[1] for data in dataset])):
        for k_idx, k in enumerate(set([data[2] for data in dataset])):
            for l_idx, l in enumerate(set([data[3] for data in dataset])):
                rows = [row for row in dataset if row[0]==i and row[1]==j and row[2]==k and row[3]==l]
                rows = sorted(rows,key=lambda x:x[-1])
                # print(i_idx,j_idx,k_idx,l_idx)
                if len(rows)>0:
                    NUM_TABLE[i_idx,j_idx,k_idx,l_idx] = rows[0][-2]


memory = []

for i in range(data_shape[0]):
    for j in range(data_shape[1]):
        for k in range(data_shape[2]):
            for l in range(data_shape[3]):
                min_val = NUM_TABLE[i,j,k,l]
                if min_val != 0:
                    if min_val != 78:
                        for val in range(100):
                            sign = np.sign(val-min_val)
                            state = (i,j,k,LEN_TABLE[l],val)
                            action = ACT_TABLE[int(sign)+1]
                            next_state = (i,j,k,LEN_TABLE[l],val-sign)
                            memory.append((state, action, -sign, next_state))
                            if sign != 0 and 0<=val+sign<=100:
                                action = ACT_TABLE[-sign+1]
                                next_state = (i,j,k,LEN_TABLE[l],val+sign)
                                memory.append((state, action, sign, next_state))
                    else:
                        for val in range(70):
                            sign = np.sign(val-min_val)
                            state = (i,j,k,LEN_TABLE[l],val)
                            action = ACT_TABLE[int(sign)+1]
                            next_state = (i,j,k,LEN_TABLE[l],val-sign)
                            memory.append((state, action, -sign, next_state))
                            if sign != 0 and 0<=val+sign<=100:
                                action = ACT_TABLE[-sign+1]
                                next_state = (i,j,k,LEN_TABLE[l],val+sign)
                                memory.append((state, action, sign, next_state))           

pickle.dump(memory, open("./source/generated_memory.pkl","wb+"))