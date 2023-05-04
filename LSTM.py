import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


class SequenceDataset(Dataset):

    def __init__(self, data_list: list):
        self.data = data_list

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.Tensor(sample[0]), torch.Tensor(sample[1])
    
    def __len__(self):
        return len(self.data)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=200, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
    

class LSTMForecaster(nn.Module):


 def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1, n_deep_layers=10, use_cuda=False, dropout=0.2):
    '''
    n_features: number of input features (1 for univariate forecasting)
    n_hidden: number of neurons in each hidden layer
    n_outputs: number of outputs to predict for each training example
    n_deep_layers: number of hidden dense layers after the lstm layer
    sequence_len: number of steps to look back at for prediction
    dropout: float (0 < dropout < 1) dropout ratio between dense layers
    '''
    super().__init__()

    self.n_lstm_layers = n_lstm_layers
    self.nhid = n_hidden
    self.use_cuda = use_cuda # set option for device selection

 # LSTM Layer
    self.lstm = nn.LSTM(n_features,
                        n_hidden,
                        num_layers=n_lstm_layers,
                        batch_first=True) # As we have transformed our data in this way
 
 # first dense after lstm
    self.fc1 = nn.Linear(n_hidden * sequence_len, n_hidden) 
 # Dropout layer 
    self.dropout = nn.Dropout(p=dropout)

 # Create fully connected layers (n_hidden x n_deep_layers)
    dnn_layers = []
    for i in range(n_deep_layers):
    # Last layer (n_hidden x n_outputs)
        if i == n_deep_layers - 1:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(self.nhid, n_outputs))
        # All other layers (n_hidden x n_hidden) with dropout option
        else:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(self.nhid, self.nhid))
        if dropout:
                dnn_layers.append(nn.Dropout(p=dropout))
        # compile DNN layers
    self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x):
    # Initialize hidden state
        hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)
        cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)

    # move hidden state to device
    # if self.use_cuda:
    #     hidden_state = hidden_state.to(device)
    #     cell_state = cell_state.to(device)
    
        self.hidden = (hidden_state, cell_state)

    # Forward Pass
        x, h = self.lstm(x, self.hidden) # LSTM
        x = self.dropout(x.contiguous().view(x.shape[0], -1)) # Flatten lstm out 
        x = self.fc1(x) # First Dense
        return self.dnn(x) # Pass forward through fully connected DNN.
    

if __name__ == "__main__":
    BATCH_SIZE = 3 # Training batch size
    split = 0.8 # Train/Test Split ratio

    sequences = create_inout_sequences([1, 2, 3, 4, 5 ,6 ,7 ,8 ,9, 10], 2)
    dataset = SequenceDataset(sequences)

    # Split the data according to our split ratio and load each subset into a
    # separate DataLoader object
    train_len = int(len(dataset)*split)
    lens = [train_len, len(dataset)-train_len]
    train_ds, test_ds = random_split(dataset, lens)
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
