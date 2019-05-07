# Author: Zhongyang Zhang
# E-mail: mirakuruyoo@gmail.com

'''
Data analysis. 
'''

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

#%%
import pandas as pd
import numpy as np                                                                                                                            
import seaborn as sns                                             
import matplotlib.pyplot as plt
sns.set()
sns.set_context("paper", font_scale=1.4)

y4 = np.array([0.061,0.0705,0.0205,0.073])                                                                                                    
y3 = np.array([0.061,0.056,0.021,0.066])                          
y2 = np.array([0.062,0.0515,0.023,0.081])                         
y1 = np.array([0.082,0.067,0.022,0.07]) 
df=pd.DataFrame(list(zip(y1,y2,y3,y4,["Coating","Tab","Tab+Coating","UBC+Coating"])), columns = ['Victim1','Victim2','Victim3','Victim4','x'])

# Set up the matplotlib figure
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
palette = "Blues"
sns.barplot(x="x", y="Victim1", data = df, palette=palette, ax=ax1)
ax1.set_title("Victim1")
ax1.set_ylabel('')    
ax1.set_xlabel('')

sns.barplot(x="x", y="Victim2", data = df, palette=palette, ax=ax2)
ax2.set_title("Victim2")
ax2.set_ylabel('')    
ax2.set_xlabel('')

sns.barplot(x="x", y="Victim3", data = df, palette=palette, ax=ax3)
ax3.set_title("Victim3")
ax3.set_ylabel('')    
ax3.set_xlabel('')

sns.barplot(x="x", y="Victim4", data = df, palette=palette, ax=ax4)
ax4.set_title("Victim4")
ax4.set_ylabel('')    
ax4.set_xlabel('')

f.savefig("ICN_results.jpg",dpi=100)
plt.show()

#%%
hist = pickle.load(open("./source/dqn_train_log.pkl",'rb'))
# sns.set_palette("violet")
res=np.array(hist)[:,-1]
f, ax = plt.subplots(figsize=(15, 15))
sns.lineplot(x='x',y='y',alpha=0.8, data=pd.DataFrame({'x':list(range(len(res))),'y':list(res)}), ax=ax)
# sns.lmplot(x='x',y='y',data=pd.DataFrame({'x':list(range(len(res))),'y':list(res)}))
ax.set_title("Average Reward in Each Exploration")
ax.set_ylabel("Average Rewards / Episode")
ax.set_xlabel('Epoch')

icns=[float(i[-3]) for i in hist]
f, ax = plt.subplots(figsize=(15, 15))
sns.lineplot(x=list(range(len(icns))),y=icns,alpha=0.8, ax=ax)
ax.set_title("Best ICN value in Each Exploration")
ax.set_ylabel("Min ICN")    
ax.set_xlabel('Epoch')