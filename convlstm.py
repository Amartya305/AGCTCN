
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 01:52:39 2021

@author: amart
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from utils import buildGraphFromJson, getPMLevelsImputed, extractComponents, countParameters, normalize, setupCUDA, saveModel
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import mean_squared_error as mse

class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],    #  B x num_features x H x W
                             self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],    #  B x num_features x H x W
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[index, ...]                                            #  B x out_channels x H x W

            combined = torch.cat((x, hx), 1)                                      #  B x (out_channels + num_features) x H x W
            gates = self.conv(combined)  # gates: S, num_features*4, H, W         
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)                                        #  B x num_features x H x W
            forgetgate = torch.sigmoid(forgetgate)                                #  B x num_features x H x W
            cellgate = torch.tanh(cellgate)                                       #  B x num_features x H x W
            outgate = torch.sigmoid(outgate)                                      #  B x num_features x H x W

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)

    


# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
'''
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=64),
        CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
        CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
    ]
]
'''

convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(48,48), input_channels=16, filter_size=5, num_features=32),
        CLSTM_cell(shape=(24,24), input_channels=32, filter_size=5, num_features=64),
        CLSTM_cell(shape=(12,12), input_channels=64, filter_size=5, num_features=64)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [32, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(12,12), input_channels=64, filter_size=5, num_features=64),
        CLSTM_cell(shape=(24,24), input_channels=64, filter_size=5, num_features=64),
        CLSTM_cell(shape=(48,48), input_channels=64, filter_size=5, num_features=32),
    ]
]

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))

class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()    # S x B x 1 x H x W
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))      # SB x 1 x H x W
        inputs = subnet(inputs)                                                 # SB x out_channels x H x W
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))        # S x B x out_channels x H x W
        outputs_stage, state_stage = rnn(inputs, None)                          # seq_len x B x out_channels x H x W
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
#        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)

class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=12)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        return inputs

class ED(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output

def getGridValues(coordsDf):
    long = np.array(coordsDf["Longitude"])
    lat = np.array(coordsDf["Latitude"])
    long = 100 * (long - 77)
    lat = 100 * (lat - 28)
    long = long.astype('int32')
    lat = lat.astype('int32')
    return long, lat

def buildGrid(X, N, T, coordsDf):
    Xout = np.zeros((T, 1, 48, 48))
    long, lat = getGridValues(coordsDf)
    gridMap = dict()
    for i in range(N):
        x = long[i] - long.min()
        y = lat[i] - lat.min()
        Xout[:, 0, y, x] = X[i, 0, :]
        gridMap[(y, x)] = i
    return Xout, gridMap

def revertGrid(X_grid, gridMap, N, T):
    Xout = np.zeros((N, 1, T))
    for (y, x) in gridMap:
        idx = gridMap[(y, x)]
        Xout[idx, 0, :] = X_grid[:, 0, y , x]
    return Xout

def split(grid, Th, Tp):
    # grid : T x 1 x 64 x 64
    T = grid.shape[0]
    start = Th
    end = Th + Tp
    X = []
    Y = []
    while (end <= T):
        x = grid[start - Th: start]
        y = grid[start : end]
        X.append(x)
        Y.append(y)
        start += Tp
        end += Tp
    return np.array(X), np.array(Y)

class  PolnData(torch.utils.data.Dataset):
    def __init__(self, Xh, Y, indices):
        self.indices = indices
        self.Xh = Xh[indices]
        self.Y = Y[indices]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.Xh[index], self.Y[index]

def train(model, Xh, Y, device, batch=32, epochs=400):
    Xh = torch.Tensor(Xh).to(device)
    Y = torch.Tensor(Y).to(device)

    indices = np.arange(Xh.shape[0])
    idxTrain, idxVal = train_test_split(indices, test_size = 0.2, random_state = 42)
    
    trainData = PolnData(Xh, Y, idxTrain)
    valData = PolnData(Xh, Y, idxVal)
    
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size = batch, shuffle = True)
    valLoader = torch.utils.data.DataLoader(valData, batch_size = batch, shuffle = False)
    
    mse = torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    
    fVloss= 0
    for e in range(epochs):
        loss = 0.0
        valLoss = 0.0
        for Xh, Y in trainLoader:
            Yp = model(Xh)
            lossT = mse(Yp, Y)
            optimizer.zero_grad()
            lossT.backward()
            optimizer.step()
            loss += lossT.item()
            
        with torch.set_grad_enabled(False):
                for Xh, Y in valLoader:
                    Yval = model(Xh)
                    lossV = mse(Yval, Y)
                    valLoss += lossV.item()
        
        epochLoss = loss / len(trainLoader)
        epochValLoss = loss/ len(valLoader)
        fVloss = epochValLoss
        print(f"Epoch: {e}, train_loss: {epochLoss}, validation_loss = {epochValLoss}")
    return model, fVloss

############################## MAIN ##############################
if __name__ == "__main__":
    #Train
    N, A, coordsDf = buildGraphFromJson('coords.json')
    trainPath = "PM10/train"
    X_train, F, T = getPMLevelsImputed(trainPath, coordsDf, N)
    X_train, scaler = normalize(X_train)
    X_train = X_train.reshape(N, 1, T)
    X_grid = buildGrid(X_train, N, T, coordsDf)
    Th = 24
    Tp = 12
    Xh, Y = split(X_grid, Th, Tp)
    
    
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).cuda()
    decoder = Decoder(convlstm_decoder_params[0], convlstm_decoder_params[1]).cuda()
    clstm = ED(encoder, decoder)
    countParameters(clstm)
    dev = setupCUDA()
    clstm.to(dev)
    
    batch = 8
    epochs = 100
    
    clstm, fVloss = train(clstm, Xh, Y, dev, batch, epochs)
    
    fn = f"ConvLSTM_Th{Th}_Tp{Tp}_b{batch}_e{epochs}_16_32_64_vl{fVloss}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
    saveModel(clstm, fn, "models", "PM10", "ConvLSTM")
    #==================================================================