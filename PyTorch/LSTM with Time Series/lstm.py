#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:05:55 2020

@author: safak
"""

import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils import clip_grad_norm


"""
EMBED_SIZE: represents the size of the input at each time step (feature)

HIDDEN_DIM: represents the size of the hidden state and cell state at each time step
(example: the hidden state and cell state will both have the shape of [3, 5, 4] if the hidden dimension is 3)

NUM_LAYERS: the number of LSTM layers stacked on top of each other 

"""

class Dataset():
    def __init__(self):
        pass
    def getDatasets(self):
        print(sns.get_dataset_names())
    def selectDataset(self,dataset_name: str):
        if not type(dataset_name) is str:
            raise TypeError("Input a valid dataset name that is string type.")
        else:
            try:
                selected = sns.load_dataset(dataset_name)
                self.df = selected
                return selected
            except:
                print("Enter a valid dataset name")
    @staticmethod
    def trainTestSplit(data,ratio = 0.8):
        if not type(data) is np.ndarray:
            raise TypeError("Only numpy ndarray types.")
        else:
            train_pct_index = int(ratio * len(data))
            training_data = data[:train_pct_index]
            test_data = data[train_pct_index:]
            return training_data, test_data
    @staticmethod
    def scaler(data,rang: tuple):
        if not type(rang) is tuple or not type(data) is np.ndarray:
            raise TypeError("Type(rang): tuple and Type(data): numpy ndarray.")
        else:
            sc = MinMaxScaler(feature_range=rang)
            normalized = sc.fit_transform(data .reshape(-1, 1))
            return normalized
    @staticmethod
    def seqModel(normalized_data,seq_dep=12):
        seq = []
        norm = len(normalized_data)
        for i in range(norm-seq_dep):
            seq_train = normalized_data[i:i+seq_dep]
            seq_label = normalized_data[i+seq_dep:i+seq_dep+1]
            seq.append((seq_train,seq_label))
        return seq
        
        
       
class LSTM(nn.Module):
    """
    - embed_size: The number of expected features in the input x

    - hidden_size: The number of features in the hidden state h
    
    - num_layers: Number of recurrent layers.
    
    - bias: default = true
    
    - batch_first – If True,  input and output shape are (batchsize, seq, feature). Default: False
    
    - dropout: Default = 0
    
    - bidirectional: Default = False
    """
    
    """ INPUTS
    input of shape: (seq_len, batchsize, input_size)
    h_0 of shape:   (num_layers * num_directions, batchsize, hidden_size)
    c_0 of shape:   (num_layers * num_directions, batchsize, hidden_size)
    """
    
    """OUTPUTS
    output of shape: (seq_len, batchsize, num_directions * hidden_size)
    h_t of shape:    (num_layers * num_directions, batchsize, hidden_size)
    c_t of shape:    (num_layers * num_directions, batchsize, hidden_size)
    """
    def __init__(self,embed_size,hidden_size,num_layers,out_size):
        super(LSTM,self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size, out_size)
        
    def forward(self,x,states):
        out, states = self.lstm(x.view(len(x) ,1, -1),states)        
        fully = self.linear(out.view(len(x), -1))
        return fully,states
        

EMBED_SIZE = 1
HIDDEN_SIZE = 100
NUM_LAYERS = 1
EPOCHS = 100


#%%

data = Dataset()
df = data.selectDataset("flights")
cols = df.columns

#%%
frames = [df[cols[0]],df[cols[1]]]
df_date = pd.concat(frames)

#%%

items = []
for i in range(df.shape[0]):
    items.append(str(df[cols[0]].iloc[i]) + " - " + list(str(df[cols[1]].iloc[i]))[0] + 
                 list(str(df[cols[1]].iloc[i]))[1] + list(str(df[cols[1]].iloc[i]))[2])
xticks = tuple(items)

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size
plt.xlabel("Months")
plt.ylabel("Total Passengers")
plt.autoscale(axis='x',tight=True)
plt.title("Plotting for Passengers over Months")
plt.grid(True)
plt.xticks(df.index,xticks,rotation='vertical',fontsize = 6)
plt.plot(df[cols[2]])
plt.show()
#%%

passengers = df[cols[2]].values.astype(np.float)
train, test = Dataset.trainTestSplit(passengers,0.9)
train_normalized = Dataset.scaler(train,(-1,1))

time_dep = 12 #there are 12 months in a year
sequence_data_normalized = Dataset.seqModel(train_normalized, time_dep)

#%%
CUDA = torch.cuda.is_available()
model = LSTM(EMBED_SIZE,HIDDEN_SIZE,NUM_LAYERS,1)

    
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

epochs = 150

        
for epochs in range(EPOCHS):
    states = (torch.zeros((1,1,HIDDEN_SIZE)),torch.zeros((1,1,HIDDEN_SIZE)))
    for seq, labels in sequence_data_normalized:
        seq = torch.from_numpy(seq)
        labels = torch.from_numpy(labels)
        seq = seq.float()
        labels = labels.float()
        out, states = model.forward(seq,states)
        loss = loss_fn(out,labels)
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    if epochs%2 == 1:
        print(f'epoch: {epochs:3} loss: {loss.item():10.8f}')
                
