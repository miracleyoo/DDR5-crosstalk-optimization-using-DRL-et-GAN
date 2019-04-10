# %%
import pickle as pkl
import os
import numpy as np
import pandas as pd
import seaborn as sns

# %%
dataset = pkl.load(open('./Datasets/matlab_direct_expanded_data_channel_comb_to10.pkl','rb'))
df = pd.DataFrame(dataset, columns=["cons","c1c2","data_rate","total_length","tab_num","icn"])

#%% [markdown]
# ## Figure1
# Plot a figure which can show the relationship between icn and tab_num, seperated by c1c2 and constraint type.

# %%
sns.set(style="darkgrid")
fig, ax = plt.subplots()
sns.lineplot(x="tab_num", y="icn",
             hue="c1c2", style="cons",
             data=df)
plt.savefig('./source/result/tab_num2icn.jpg', format='jpg', dpi=1000)

#%% [markdown]
# ## Figure2
# Plot a figure which can show the relationship between icn and data_rate, seperated by c1c2 and constraint type.

# %%
sns.set(style="darkgrid")
fig, ax = plt.subplots()
sns.lineplot(x="data_rate", y="icn",
             hue="c1c2", style="cons",
             data=df)
plt.savefig('./source/result/datarate2icn.jpg', format='jpg', dpi=1000)

#%% [markdown]
# ## Figure3
# Plot a figure which can show the relationship between icn and data_rate, seperated by c1c2 and constraint type.

# %%
sns.set(style="darkgrid")
fig, ax = plt.subplots()
sns.lineplot(x="total_length", y="icn",
             hue="c1c2", style="cons",
             data=df)
plt.savefig('./source/result/total_length2icn.jpg', format='jpg', dpi=1000)

#%% [markdown]
# 尝试使用lightgbm

#%%
import lightgbm as lgb
train_data = lgb.Dataset([i[:-1] for i in dataset], label=[i[-1] for i in dataset])
Y = df[-1]
X = df_train.drop(-1, axis=1)
lgb_train = lgb.Dataset(X, Y)
validation_data = train_data.create_valid('validation.svm')

#%%
def getxy(res):
  data=[i for i in dataset if i[0]==res[0] and i[1]==res[1] and i[2]==res[2] and i[3]==res[3]]
  data.sort(key=lambda x: x[-2])
  xs = [i[-2] for i in data]
  ys = [i[-1] for i in data]
  return xs,ys

#%%
prefix = "./source/num_icn/"
for i_idx, i in enumerate(sorted(set([data[0] for data in dataset]))):
    for j_idx, j in enumerate(sorted(set([data[1] for data in dataset]))):
        for k_idx, k in enumerate(sorted(set([data[2] for data in dataset]))):
            for l_idx, l in enumerate(sorted(set([data[3] for data in dataset]))):
                xs, ys = getxy([i,j,k,l])
                plt.cla()
                ll = plt.plot(xs, ys)
                plt.setp(ll, markersize=5)
                plt.setp(ll, markerfacecolor='C0')
                # plt.show()
                fname = prefix+"{}_{}_{}_{}.jpg".format(i,j,k,l)
                plt.savefig(fname, dpi=100, facecolor='w', edgecolor='w')