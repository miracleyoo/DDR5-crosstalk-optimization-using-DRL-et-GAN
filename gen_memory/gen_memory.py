import numpy as np
import pickle

NUM_TABLE = np.array([[[[50,50,75,100],[50,57,75,100],[100,100,100,100]],
[[19,38,50,75],[50,50,57,75],[75,75,100,100]]],
[[[25,38,50,100],[25,50,75,75],[100,100,100,100]],
[[19,25,50,75],[25,50,75,100],[75,100,100,100]]]])

LEN_TABLE = np.array([1500,2000,3000,4000])
ACT_TABLE = np.array([4,0,3])

memory = []

for i in range(2):
    for j in range(2):
        for k in range(3):
            for l in range(4):
                min_val = NUM_TABLE[i,j,k,l]
                for val in range(100):
                    sign = np.sign(val-min_val)
                    state = (i,j,k,LEN_TABLE[l],val)
                    action = ACT_TABLE[sign+1]
                    next_state = (i,j,k,LEN_TABLE[l],val-sign)
                    memory.append((state, action, -sign, next_state))
                    if sign != 0 and 0<=val+sign<=100:
                        action = ACT_TABLE[-sign+1]
                        next_state = (i,j,k,LEN_TABLE[l],val+sign)
                        memory.append((state, action, sign, next_state))

pickle.dump(memory, open("./source/generated_memory.pkl","wb+"))