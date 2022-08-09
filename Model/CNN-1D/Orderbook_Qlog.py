import pandas as pd
import numpy as np
import itertools
#from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import scipy.stats
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from hyperopt.pyll.base import scope
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

print("DONE")

sns.set()

#format the name of the file wanted for the simulation using the two var asset and level
asset = 'AAPL'
level = 10 
data = pd.read_csv('../../Data/{0}_2012-06-21_34200000_57600000_orderbook_{1}.csv'.format(asset, level), header=None)
data.head()

levels = list(range(1, level + 1))
print(levels)
iters = [iter(levels), iter(levels), iter(levels), iter(levels)]
print(iters)

abv = ['ask', 'volume_ask', 'bid', 'volume_bid'] * level
print(abv)
nums = [x for x in itertools.chain.from_iterable(itertools.zip_longest(levels, levels, levels, levels)) if x]
data.columns = list(map(lambda x, y: '{0}_{1}'.format(x, y), abv, nums))

data.head()

#Assest = Stock name 
#Level = order book level

def load_data(asset, level):
    data = pd.read_csv('../../Data/{0}_2012-06-21_34200000_57600000_orderbook_{1}.csv'.format(asset, level), header=None)
    
    levels = list(range(1, level + 1))
    iters = [iter(levels), iter(levels), iter(levels), iter(levels)]
    abv = ['ask', 'volume_ask', 'bid', 'volume_bid'] * level
    nums = [x for x in itertools.chain.from_iterable(itertools.zip_longest(levels, levels, levels, levels)) if x]
    data.columns = list(map(lambda x, y: '{0}_{1}'.format(x, y), abv, nums))
    
    return data




# commented out the graph plotting 

assets = ['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']
#fig, axs = plt.subplots(len(assets), 2, figsize = (15, 15))
#colours = cm.prism(np.linspace(0, 1, len(assets)))  #change colourmap

#print(fig)

#for i, (ax, colour) in enumerate(zip(axs, colours)): #zip- Iterate over two lists in parallel axis colours
for i in range(len(assets)):
    asset = assets[i]
    source_data = load_data(asset, 1)
    diff_volumes = source_data.volume_ask_1 - source_data.volume_bid_1
    mid_price = source_data.ask_1 - source_data.bid_1
    
    # ax[0].hist(mid_price[~np.isnan(mid_price)], bins = 30)
    # ax[0].set_title('Mid-price for ' + asset)
    # ax[0].set_xlabel('USD')
    # ax[0].set_ylabel('Dist')
    
    # ax[1].hist(diff_volumes[~np.isnan(diff_volumes)], bins = 30)
    # ax[1].set_title('Difference of ask-bid volumes for ' + asset)
    # ax[1].set_xlabel('Volume difference')
    # ax[1].set_ylabel('Dist')
    
#plt.tight_layout()


# reading csv files

data1 = pd.read_csv('../../Data/AAPL_2012-06-21_34200000_57600000_orderbook_1.csv', header=None)
data2 = pd.read_csv('../../Data/AAPL_2012-06-21_34200000_57600000_message_1.csv', header=None)
# displaying input files
print("data1 ", data1.shape)
#print("data1 info", data1.info())
print(data1.head())
print("data2 ", data2.shape)
#print("data2 info", data2.info())
print(data2.head())

data1.columns=["ask", "volume_ask", "bid", "volume_bid"]
data2.columns=["Time", "Type", "OrderID", "Size", "Price", "Direction"]



#Change the output format to be between 0.1 & 0.9 instead of -1 & 1


# using merge function by setting how='outer'
result = data1.merge(data2, left_index=True, right_index=True, how='left')
result['Direction'].replace({-1 : 0.1, 1 : 0.9}, inplace=True)
print("RESULTS")  
# displaying result
print("Shape of the result matrix is = ",result.shape)
#print(result.head())
result.to_csv("combine_resutls.csv", encoding='utf-8', index=False)


cols = ["ask", "volume_ask", "bid", "volume_bid", "Type", "Size", "Price", "Direction"]
result = result[cols]
#sns.countplot(result.ask(100))
#result.info
# #of rows
print("number of rows =",result.shape[0])
print("number of columns =",result.shape[1])
print("Number of inputs for each categories")
result.Type.value_counts()/result.shape[0]

RANDOM_SEED = 42
#Creating the 7 Inputs for the DNN 
X = result[["ask", "volume_ask", "bid", "volume_bid", "Type", "Size", "Price"]]
print("Shape of the matrix X is:\t",X.shape)
#print(X.head())

#Creating the 1 output for the DNN to train the network with results 
Y = result[["Direction"]]
print("Shape of the matrix Y is:\t",Y.shape,"\n")
# print(Y.info())
#print(Y.head())



##pre-processing

print(X)
print(Y)

print(X.head(1))

def f_default():
	return X 

## normalise with norm_x=(x-mean_x)/(std_x)

def norm(X):
	return (X-X.mean())/X.std()

## minimal scaling
def minmax(X):
	return (X-X.min())/(X.max()-X.min())

## normalise with max
def max(X):
	return X/X.max()

## log return
def log_return(X):
	return X


def preprocess(case):
	return{
    		1:norm,
		2:minmax,
		3:max,
		4:log_return,
	
	}.get(case,f_default)

#print("fea 1")
#print(preprocess(3)(X))
#print("norm x")
#print(max(X))
#print("X")
#print(X)

X=preprocess(3)(X)

#Split the data between trianing 80% and test 20% no shuffling as it is a time series
#0.2 = 20% of the data for test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(X, Y, test_size=0.2, shuffle = False, stratify = None)
print("Shape of the matrix X_train_np is:\t",X_train_np.shape)
print("Shape of the matrix X_test_np is:\t",X_test_np.shape,"\n")
print("Shape of the matrix Y_train_np is:\t",Y_train_np.shape)
print("Shape of the matrix Y_test_np is:\t",Y_test_np.shape,"\n")


#checking the X_test_np data
X_test_np.head()


#checking the Y_test_np data
Y_test_np.head()




np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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
    #print("x in fc3")
    #print(x)
    #print(self.fc3(x))
    #print(torch.sigmoid(self.fc3(x)))
    return torch.sigmoid(self.fc3(x))*0.8+0.1

net = Net(X_train_T.shape[1])
print(X_train_T.shape[1])
print("NN structure is: ",net)
print(net.parameters)

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
    return (torch.sigmoid(self.fc3(x))*0.8+0.1)

net1 = Net1(X_train_T.shape[1])
print(X_train_T.shape[1])
print("NN structure is: ",net1)
print(net1.parameters)
#t = torch.randn(4)
#print(t)
#print(torch.sigmoid(t))


#quantized network
class QuantWeightNet(nn.Module):
    def __init__(self, n_features):
        super(QuantWeightNet, self).__init__()
        self.fc1   = qnn.QuantLinear(n_features, 7, bias=True, weight_bit_width=3)
        self.relu1 = nn.ReLU()
        self.fc2   = qnn.QuantLinear(7, 20, bias=True, weight_bit_width=3)
        self.relu2 = nn.ReLU()
        self.fc4   = qnn.QuantLinear(20, 20, bias=True, weight_bit_width=3)
        self.relu3 = nn.ReLU()
        self.fc5   = qnn.QuantLinear(20, 20, bias=True, weight_bit_width=3)
        self.relu4 = nn.ReLU()
        self.fc3   = qnn.QuantLinear(20, 1, bias=False, weight_bit_width=3)
        

    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        out = self.relu3(self.fc4(out))
        #out = self.relu4(self.fc5(out))
        return (torch.sigmoid(self.fc3(out))*0.8+0.1)

Net_q = QuantWeightNet(X_train_T.shape[1])
print("NN structure is: ",Net_q)
print(Net_q.parameters)

criterion = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')


net = Net(X_train_T.shape[1])
net1 = Net1(X_train_T.shape[1])
net_q= QuantWeightNet(X_train_T.shape[1])
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer1 = optim.Adam(net1.parameters(), lr=0.0003)
optimizer_q = optim.Adam(net_q.parameters(), lr=0.0003)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#transfer data to the GPU
X_train = X_train_T.to(device)
Y_train = Y_train_T.to(device)
X_test = X_test_T.to(device)
Y_test = Y_test_T.to(device)
net = net.to(device)
criterion = criterion.to(device)

net1 = net1.to(device)
net_q = net_q.to(device)

device
#print(list(net.parameters()))
#print(optimizer.param_groups)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#Input Normalization

#print(X_train[1,:])
#X_test[:,[0,2,6]]=X_test[:,[0,2,6]]/X_train[0,0]
#X_train[:,[0,2,6]]=X_train[:,[0,2,6]]/X_train[0,0]
#X_test[:,[1,3,5]]=X_test[:,[1,3,5]]/X_train[0,1]
#X_train[:,[1,3,5]]=X_train[:,[1,3,5]]/X_train[0,1]
#X_test[:,[1,3,5]]=X_test[:,[1,3,5]]/X_train[0,1]
#X_test[:,4]=X_test[:,4]/X_train[0,4]
#X_train[:,4]=X_train[:,4]/X_train[0,4]
#print(X_test[:,[0,2,6]])
#print(X_train[:,[0,2,6]])
#print(X_test)
#print(X_train)


def calculate_accuracy(Y_true, Y_pred):
  predicted = Y_pred.ge(.5).view(-1) ##threshold 0.5
  #print("prediected")
  #print(predicted)
  #predicted1 = Y_pred.ge(.5)
  #print(predicted1*0.8+0.1)  
  return (Y_true == (predicted*0.8+0.1)).sum().float() / len(Y_true)

def round_tensor(t, decimal_places=5):
  return round(t.item(), decimal_places)

#print(net.fc1.weight)

#print(net.fc2.weight)

#print(net.fc3.weight)
#print(Y_pred)
#print(Y_train)

print(optimizer1.param_groups)
print(optimizer_q.param_groups)

for epoch in range(1000):
      optimizer1.zero_grad() #reset Gradient descent
      optimizer_q.zero_grad()      


      Y_pred = net1(X_train)
      Y_pred_q = net_q(X_train)

      #print("pred Y")
      #print(Y_pred)
      #print("train Y")
      #print(Y_train)
      Y_pred = torch.squeeze(Y_pred)
      Y_pred_q = torch.squeeze(Y_pred_q)
      criterion=nn.BCELoss()
      train_loss = criterion(Y_pred, Y_train)
      train_loss_q = criterion(Y_pred_q, Y_train)
      #train_loss = nn.BCELoss(Y_pred, Y_train)
      #train_loss.retain_grad()
      #print(train_loss)
      train_loss.backward() #propagate the error currently making     

      #print(net.fc1.weight.grad)
      #print(net.fc2.weight.grad)
      #print(net.fc3.weight.grad)

      optimizer1.step()      #optimise 

      train_loss_q.backward() #propagate the error currently making     
      optimizer_q.step()

      #print(optimizer.param_groups)
      #print(net.fc1.weight)
      if epoch % 100==0:
        train_acc = calculate_accuracy(Y_train, Y_pred)
        train_acc_q = calculate_accuracy(Y_train, Y_pred)
        #print(net.fc1.weight)
        #print(net.fc1.weight.grad)
        print(Y_pred)
        print(Y_train)
        print(train_loss)
        print(Y_pred_q)
        print(Y_train)
        print(train_loss_q)
        Y_test_pred = net1(X_test)
        Y_test_pred = torch.squeeze(Y_test_pred)
        test_loss = criterion(Y_test_pred, Y_test)
        test_acc = calculate_accuracy(Y_test, Y_test_pred)
        print(
  f'''epoch {epoch}
  Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
  Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
  ''')
        Y_test_pred_q = net_q(X_test)
        Y_test_pred_q = torch.squeeze(Y_test_pred_q)
        test_loss_q = criterion(Y_test_pred_q, Y_test)
        test_acc_q = calculate_accuracy(Y_test, Y_test_pred_q)
        print(
  f'''epoch {epoch}
  QTrain set - loss: {round_tensor(train_loss_q)}, accuracy: {round_tensor(train_acc_q)}
  QTest  set - loss: {round_tensor(test_loss_q)}, accuracy: {round_tensor(test_acc_q)}
  ''')
      #print(net.fc1.weight)
      #print(net.fc1.weight.grad)
      #print(net.fc1.bias)
      #print(net.fc1.bias.grad)
      #print(train_loss)

    #if epoch % 100 == 0:
      #print(net.fc1.weight)
      #print(np.nonzero(net.fc1.weight.grad))
      #print(net.fc1.bias)
      #print(np.nonzero(net.fc1.bias.grad))

print(optimizer1.param_groups)
print(optimizer_q.param_groups)

#saving model

MODEL_PATH = 'model1.pt'
torch.save(net1, MODEL_PATH)
net1 = torch.load(MODEL_PATH)

print("Done saving net")

MODEL_PATH = 'model_q.pt'
torch.save(net_q.state_dict(), MODEL_PATH)
net_q = torch.load(MODEL_PATH)
print("Done saving Q net")




#classes = ['buy', 'sell']
#y_pred = net1(X_test)
#y_pred = y_pred.ge(.5).view(-1).cpu()
#Y_test = Y_test.cpu()
#print(classification_report(Y_test, y_pred, target_names=classes))



