# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 20:38:56 2021

@author: Amartya Choudhury
"""
from utils import buildGraphFromJson, getPMLevelsImputed, countParameters, setupCUDA, normalize, saveModel
import torch.nn as nn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from visuals import plotPredictions
from sklearn.metrics import mean_squared_error as mse

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, output_seq):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers
        self.output_seq = output_seq
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda').requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda').requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -self.output_seq : , :].contiguous())
        return out

def createSequences(data, Td, Tp, N, T):
    if(len(data.shape) != 2):
        raise ValueError('expected a 2-D or more input array')
    X = np.array([])
    Y = np.array([])

    for n in range(N):
        start = Td
        while (start < T):
            if (start + Tp > T):
                break
            x = data[n, start - Td : start]      # Td
            x = np.reshape(x, (1, Td))           # 1 x Td
            y = data[n, start : start + Tp,]     # Tp
            y = np.reshape(y, (1, Tp))           # 1 x  Tp
            if (X.shape[0] == 0):
                X = x
                Y = y
            else:
                X = np.vstack((X, x))
                Y = np.vstack((Y, y))
            start += Tp
    return X, Y

class LSTMData(torch.utils.data.Dataset):
    def __init__(self, X, Y, indices):
        self.indices = indices
        self.X = X[indices]
        self.Y = Y[indices]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.X[index], self.Y[index]

def trainLSTM(model, X, Y, device, batch=32, epochs=400):
    X = torch.Tensor(X).to(device)
    Y = torch.Tensor(Y).to(device)

    indices = np.arange(X.shape[0])
    idxTrain, idxVal = train_test_split(indices, test_size = 0.2, random_state = 42)
    
    trainData = LSTMData(X, Y, idxTrain)
    valData = LSTMData(X, Y, idxVal)
    
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size = batch, shuffle = True)
    valLoader = torch.utils.data.DataLoader(valData, batch_size = batch, shuffle = False)
    
    mse = torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.005)
    
    fVloss= 0
    for e in range(epochs):
        loss = 0.0
        valLoss = 0.0
        for Xb, Yb in trainLoader:
            Yp = model(Xb)
            lossT = mse(Yp, Yb)
            optimizer.zero_grad()
            lossT.backward()
            optimizer.step()
            loss += lossT.item()
  
        with torch.set_grad_enabled(False):
                for Xb, Yb in valLoader:
                    Yval = model(Xb)
                    lossV = mse(Yval, Yb)
                    valLoss += lossV.item()
        
        epochLoss = loss / len(trainLoader)
        epochValLoss = loss/ len(valLoader)
        fVloss = epochValLoss
        print(f"Epoch: {e}, train_loss: {epochLoss}, validation_loss = {epochValLoss}")
        
    return model, fVloss

################################### MAIN ########################################
if __name__ == "__main__":
    N, A, coordsDf = buildGraphFromJson('coords.json')
    '''
    trainPath = "PM10/train"
    X_train, F, T = getPMLevelsImputed(trainPath, coordsDf, N)
    X_train, scaler = normalize(X_train)
    
    Td = 24
    Tp = 12
    X, Y = createSequences(X_train, Td, Tp, N, T)
    X = X.reshape(-1, Td, 1)
    Y = Y.reshape(-1, Tp, 1)
    
    epochs = 400
    batch = 128
    
    device = setupCUDA()
    model = LSTM(1, 32, 2, 1, Tp)
    model.to(device)
    
    countParameters(model)
    
    #trainLSTM(model, X, Y, device, batch, epochs)
    
    fn = f"Lstm_Td{Td}_Tp{Tp}_32_2_b{batch}_e{epochs}_vl{0.023}_" + datetime.now().strftime("%H_%M_%S_%d-%m-%y")+".pt"
    saveModel(model, fn, "models", "PM10", "LSTM")
    '''
    
    testPath = "PM10/test"
    X_test, F, T = getPMLevelsImputed(testPath, coordsDf, N)
    
    X_test, scaler = normalize(X_test)
    
    Td = 24
    Tp = 12
    Xt, Yt = createSequences(X_test, Td, Tp, N, T)
    
    
    path = "models/PM10/LSTM/" + "Lstm_Td24_Tp12_32_2_b128_e400_vl0.023_00_36_02_12-07-21.pt"
    device = setupCUDA()
    lstm = LSTM(1, 32, 2, 1, Tp)
    lstm.to(device)
    countParameters(lstm)
    lstm.load_state_dict(torch.load(path))
    
    Xt = torch.Tensor(Xt).to(device)
    Xt = Xt.reshape(-1, Td, 1)
    
    
    Yp = lstm(Xt).cpu().detach().numpy()
    
    Yp = Yp.reshape(-1, 1)
    Yp = scaler.inverse_transform(Yp)
    Yt = Yt.reshape(-1, 1)
    Yt = scaler.inverse_transform(Yt)
    
    Yp = Yp.reshape(N, 1, -1)
    Yt = Yt.reshape(N, 1, -1)
    
    
    plotPredictions(Yt[:, :, 0:500], Yp[:, :, 0:500], coordsDf, N)
    mse(Yt[:, 0, :], Yp[:, 0, :])