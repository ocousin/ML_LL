import pandas as pd
import numpy as np
import itertools
#from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import scipy.stats
#from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
#from hyperopt.pyll.base import scope
import seaborn as sns
from matplotlib import cm, pyplot as plt
from scipy import stats as st
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
import torch.nn.functional as F
import torch
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant
from brevitas.quant import Int8BiasPerTensorFixedPointInternalScaling as BiasQuant
from brevitas.export import PyXIRManager
from brevitas.export import FINNManager
from preProcessing import *


RANDOM_SEED = 42
## reading csv files and format the data adding column names
## set Asset and level 
asset = 'AAPL'
level = 1     # Can only be 1, 5 and 10

data1 = loadDataOrderBook('../../Data/{0}_2012-06-21_34200000_57600000_orderbook_{1}.csv', asset, level)
data2 = loadDataMessage('../../Data/{0}_2012-06-21_34200000_57600000_message_{1}.csv', asset, level)

# displaying input files
print("ORDER BOOK")
print("data1.shape = ", data1.shape)
print(data1.head())

print("MESSAGES")
print("data2.shape = ", data1.shape)
print(data2.head())

data1.columns=["ask", "volume_ask", "bid", "volume_bid"]
data2.columns=["Time", "Type", "OrderID", "Size", "Price", "Direction"]


#Change the output format to be between 0.1 & 0.9 instead of -1 & 1
result_high=0.9
result_low=1-result_high

# using merge function by setting how='outer'
result = data1.merge(data2, left_index=True, right_index=True, how='left')
result['Direction'].replace({-1 : result_low, 1 : result_high}, inplace=True)

cols = ["ask", "volume_ask", "bid", "volume_bid", "Type", "Size", "Price", "Direction"]
result = result[cols]

print("RESULTS")  
# displaying result
print("Shape of the result matrix is = ",result.shape)
print(result.head())

print("number of rows =",result.shape[0])
print("number of columns =",result.shape[1])
print("Number of inputs for each categories")
#result.Type_1.value_counts()/result.shape[0]

#Creating the 7 Inputs for the DNN 
X = result[["ask", "volume_ask", "bid", "volume_bid", "Type", "Size", "Price"]]
print("Shape of the matrix X is:\t",X.shape)

#Creating the 1 output for the DNN to train the network with results 
Y = result[["Direction"]]
print("Shape of the matrix Y is:\t",Y.shape,"\n")

##pre-processing
print("INPUT VALUES TO FEED THE CNN")
print(X.head(1))
print("OUTPUT VALUES FOR TRAINING")
print(Y.head(1))

#Normalise the INPUT value to improve the learning
X=preprocess(1)(X)


#Split the data between trianing 80% and test 20% no shuffling as it is a time series
#0.2 = 20% of the data for test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(X, Y, test_size=0.2, shuffle = False, stratify = None)
print("Shape of the matrix X_train_np is:\t",X_train_np.shape)
print("Shape of the matrix Y_train_np is:\t",Y_train_np.shape)
print("Shape of the matrix X_test_np is:\t",X_test_np.shape,"\n")
print("Shape of the matrix Y_test_np is:\t",Y_test_np.shape,"\n")

#checking the X_test_np data
print("INPUT",X_test_np.head(1))

#checking the Y_test_np data
print("OUTPUT",Y_test_np.head(1))

print('Convert Numpy array to Torch\n')
#_T for Torch arrays
X_train_T = torch.from_numpy(X_train_np.to_numpy()).float()

print("Dimension of input tensor:", X_train_T.dim())
print("Input tensor Size:\n",X_train_T.size())


#remove a dimension
Y_train_T = torch.squeeze(torch.from_numpy(Y_train_np.to_numpy()).float())
X_test_T = torch.from_numpy(X_test_np.to_numpy()).float()
Y_test_T = torch.squeeze(torch.from_numpy(Y_test_np.to_numpy()).float())

print("Dimension of input tensor:", Y_train_T.dim())
print("Input tensor Size:\n",Y_train_T.size())
print("Shape of the matrix X_train_T is:\t",X_train_T.shape)
print("Shape of the matrix X_test_T is:\t",X_test_T.shape,"\n")
print("Shape of the matrix Y_train_T is:\t",Y_train_T.shape)
print("Shape of the matrix Y_test_T is:\t",Y_test_T.shape,"\n")


#base network
class Net(nn.Module):
  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 7)
    self.fc2 = nn.Linear(7, 20)
    self.fc3 = nn.Linear(20, 1)
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))*(result_high-result_low)+result_low

net = Net(X_train_T.shape[1])
#print(X_train_T.shape[1])
#print("NN structure is: ",net)
#print(net.parameters)

#augmented network
class Net1(nn.Module):
  def __init__(self, n_features):
    super(Net1, self).__init__()
    self.fc1 = nn.Linear(n_features, 7)
    self.fc2 = nn.Linear(7, 20)
    self.fc4 = nn.Linear(20, 20)
    self.fc5 = nn.Linear(20, 20)
    self.fc3 = nn.Linear(20, 1)
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc4(x))
    #x = F.relu(self.fc5(x))
    return (torch.sigmoid(self.fc3(x))*(result_high-result_low)+result_low)

net1 = Net1(X_train_T.shape[1])
#print(X_train_T.shape[1])
#print("NN structure is: ",net1)
#print(net1.parameters)

q_bit_width=16

#quantized network
#see examples https://github.com/Xilinx/brevitas
class QuantWeightNet(nn.Module):
    def __init__(self, n_features):
        super(QuantWeightNet, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=16, return_quant_tensor=True)
        self.fc1   = qnn.QuantLinear(n_features, 7, bias=True, weight_bit_width=q_bit_width)
        self.relu1 = qnn.QuantReLU()
        self.fc2   = qnn.QuantLinear(7, 20, bias=True, weight_bit_width=q_bit_width)
        self.relu2 = qnn.QuantReLU()
        self.fc4   = qnn.QuantLinear(20, 20, bias=True, weight_bit_width=q_bit_width)
        self.relu3 = qnn.QuantReLU()
        self.fc5   = qnn.QuantLinear(20, 20, bias=True, weight_bit_width=q_bit_width)
        self.relu4 = qnn.QuantReLU()
        self.fc3   = qnn.QuantLinear(20, 1, bias=False, weight_bit_width=q_bit_width)
        

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc4(x))
        x = self.relu4(self.fc5(x))
        return (torch.sigmoid(self.fc3(x))*(result_high-result_low)+result_low)

        #return ((self.fc3(x))*(result_high-result_low)+result_low) #Why is this stop ONNX export to work tested with 1 Layer


net_q= QuantWeightNet(X_train_T.shape[1])
#print(X_train_T.shape[1])
#print("NN structure is: ",net_q)
#print(net_q.parameters)    
    

#Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities
criterion = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')

#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer1 = optim.Adam(net1.parameters(), lr=0.0003)
optimizer_q = optim.Adam(net_q.parameters(), lr=0.0003)



#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #eport ONNX does not work with GPU data
device='cpu'
#transfer data to the GPU
X_train = X_train_T.to(device)
Y_train = Y_train_T.to(device)
X_test = X_test_T.to(device)
Y_test = Y_test_T.to(device)

criterion = criterion.to(device)

net = net.to(device)
net1 = net1.to(device)
net_q = net_q.to(device)

device
#print(list(net.parameters()))
#print(optimizer.param_groups)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#print(optimizer1.param_groups)
#print(optimizer_q.param_groups)


MODEL_PATH = 'model1.pt'    # Full FP implementation 
torch.save(net1, MODEL_PATH)
#net1 = torch.load(MODEL_PATH) #no need to re-load
print("Done saving net")

MODEL_PATH = 'model_q.pt'   #Quantized version of the full FP implemtation
torch.save(net_q.state_dict(), MODEL_PATH)
#net_q = torch.load(MODEL_PATH) #no need to re-load
print("Done saving Q net")


#inp=torch.empty((1,1,1,7))
inp=torch.randn(1,1,1,7)
#inp=torch.randn(1,7)

#print(inp)
#print(inp.shape)

FINNManager.export(net_q, input_t=inp, export_path='finn_QWNet111.onnx')



for epoch in range(1000):
      optimizer1.zero_grad() #reset Gradient descent
      optimizer_q.zero_grad()      
      Y_pred = net1(X_train)
      Y_pred_q = net_q(X_train)
      Y_pred = torch.squeeze(Y_pred)
      Y_pred_q = torch.squeeze(Y_pred_q)
      criterion=nn.BCELoss()
      train_loss = criterion(Y_pred, Y_train)
      train_loss_q = criterion(Y_pred_q, Y_train)
      train_loss.backward() #propagate the error currently making     
      optimizer1.step()      #optimise 
      train_loss_q.backward() #propagate the error currently making     
      optimizer_q.step()

      if epoch % 100==0:
        train_acc = calculate_accuracy(Y_train, Y_pred,result_high,result_low)
        train_acc_q = calculate_accuracy(Y_train, Y_pred,result_high,result_low)
        Y_test_pred = net1(X_test)
        Y_test_pred = torch.squeeze(Y_test_pred)
        test_loss = criterion(Y_test_pred, Y_test)
        test_acc = calculate_accuracy(Y_test, Y_test_pred,result_high,result_low)
        print(
  f'''epoch {epoch}
  Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
  Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
  ''')
        Y_test_pred_q = net_q(X_test)
        Y_test_pred_q = torch.squeeze(Y_test_pred_q)
        test_loss_q = criterion(Y_test_pred_q, Y_test)
        test_acc_q = calculate_accuracy(Y_test, Y_test_pred_q,result_high,result_low)
        print(
  f'''epoch {epoch}
  QTrain set - loss: {round_tensor(train_loss_q)}, accuracy: {round_tensor(train_acc_q)}
  QTest  set - loss: {round_tensor(test_loss_q)}, accuracy: {round_tensor(test_acc_q)}
  ''')
