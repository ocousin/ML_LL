
import pandas as pd


# data1 = loadDataOrderBook('../../Data/{0}_2012-06-21_34200000_57600000_orderbook_{1}.csv', asset, level)
# data2 = loadDataMessage('../../Data/{0}_2012-06-21_34200000_57600000_message_{1}.csv', asset, level)



def  loadDataOrderBook (filename, asset, level):
	if level not in [1, 5, 10]:
		raise SyntaxError('Files only supports level 1, 5 and 10')
	return pd.read_csv(filename.format(asset, level))


def loadDataMessage (filename, asset, level):
	if level not in [1, 5, 10]:
		raise SyntaxError('Files only supports level 1, 5 and 10')
	return pd.read_csv(filename.format(asset, level))




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

def calculate_accuracy(Y_true, Y_pred):
  predicted = Y_pred.ge(.5).view(-1) ##threshold 0.5
  #print("prediected")
  #print(predicted)
  #predicted1 = Y_pred.ge(.5)
  #print(predicted1*0.8+0.1)  
  return (Y_true == (predicted*0.8+0.1)).sum().float() / len(Y_true)

def round_tensor(t, decimal_places=5):
  return round(t.item(), decimal_places)