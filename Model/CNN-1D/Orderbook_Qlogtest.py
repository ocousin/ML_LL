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

data1 = pd.read_csv('../../Data/AMZN_2012-06-21_34200000_57600000_orderbook_1.csv', header=None)
data2 = pd.read_csv('../../Data/AMZN_2012-06-21_34200000_57600000_message_1.csv', header=None)
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
	
	mean_x=np.mean(X,0)
	std_x=np.std(X,0)

	#print("mean of X")
	#print(np.mean(X,0))
	#print(mean_x)


	#print("std of X")
	#print(np.std(X,0))
	#print(std_x)


	#print("norm of X");
	norm_x=((X-mean_x)/std_x)

	#print((X-mean_x)/std_x)
	X=norm_x

	return (X-X.mean())/X.std()

## minimal scaling
def minmax(X):
	return (X-X.min())/(X.max()-X.min())

## normalise with max
def max(X):
	return X/X.max()


## log return
def lor_return(X):
	return X


def preprocess(case):
	return{
    		1:norm,
		2:minmax,
		3:max,
	
	}.get(case,f_default)

print("fea 1")
print(preprocess(3)(X))
print("norm x")
print(max(X))
print("X")
print(X)

#Split the data between trianing80% and test 20% no shuffling as it is a time series
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




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#transfer data to the GPU
X_train = X_train_T.to(device)
Y_train = Y_train_T.to(device)
X_test = X_test_T.to(device)
Y_test = Y_test_T.to(device)
#net = net.to(device)
#criterion = criterion.to(device)

#net1 = net1.to(device)
#net_q = net_q.to(device)

device
#print(list(net.parameters()))
#print(optimizer.param_groups)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(X_test)
print(X_train)



criterion = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
print("training loss")

a=torch.from_numpy(0.1*np.ones(1, dtype=float))
b=torch.from_numpy(0.9*np.ones(1, dtype=float))
a1=torch.from_numpy(0.1*np.ones(100, dtype=float))
b1=torch.from_numpy(0.9*np.ones(100, dtype=float))
a2=torch.from_numpy(0.1*np.ones(2, dtype=float))
b2=torch.from_numpy(0.9*np.ones(2, dtype=float))

ab=torch.from_numpy(np.array([0.1, 0.9]))
ba=torch.from_numpy(np.array([0.9, 0.1]))
print(ab)
print(ba)

print("1 element")
print(criterion(a,a))
print(criterion(a,b))
print(criterion(b,a))
print(criterion(b,b))


print("2 element")
print(criterion(a2,a2))
print(criterion(a2,b2))
print(criterion(b2,b2))
print(criterion(b2,b2))

print("mix element")
print(criterion(ab,ba))
print(criterion(ba,ab))
print(criterion(ab,a2))
print(criterion(ba,b2))

print(criterion(ab,ab))
print(criterion(ba,ba))

print("100 element")
print(criterion(a1,a1))
print(criterion(a1,b1))
print(criterion(b1,a1))
print(criterion(b1,b1))


