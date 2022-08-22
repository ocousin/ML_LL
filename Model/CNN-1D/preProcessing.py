
import pandas as pd


# data1 = loadDataOrderBook('../../Data/{0}_2012-06-21_34200000_57600000_orderbook_{1}.csv', asset, level)
# data2 = loadDataMessage('../../Data/{0}_2012-06-21_34200000_57600000_message_{1}.csv', asset, level)



def loadDataOrderBook (filename, asset, level):
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
def log_return(X, level, num_lags):
    out_X = pd.DataFrame()
    print(out_X.head())
    for i in range(1, level + 1):
        out_X['log_return_ask_{0}'.format(i)] = np.log(X['ask_{0}'.format(i)].pct_change() + 1)

        out_X['log_return_bid_{0}'.format(i)] = np.log(X['bid_{0}'.format(i)].pct_change() + 1)

        out_X['log_ask_{0}_div_bid_{0}'.format(i)] = np.log(X['ask_{0}'.format(i)] / X['bid_{0}'.format(i)])

        out_X['log_volume_ask_{0}_div_bid_{0}'.format(i)] = np.log(X['volume_ask_{0}'.format(i)] / X['volume_bid_{0}'.format(i)])
        
        out_X['log_volume_ask_{0}'.format(i)] = np.log(X['volume_ask_{0}'.format(i)])
        
        out_X['log_volume_bid_{0}'.format(i)] = np.log(X['volume_bid_{0}'.format(i)])
        
        if i != 1:
            out_X['log_ask_{0}_div_ask_1'.format(i)] = np.log(X['ask_{0}'.format(i)] / X['ask_1'])
            out_X['log_bid_{0}_div_bid_1'.format(i)] = np.log(X['bid_{0}'.format(i)] / X['bid_1'])
            out_X['log_volume_ask_{0}_div_ask_1'.format(i)] = np.log(X['volume_ask_{0}'.format(i)] / X['volume_ask_1'])
            out_X['log_volume_bid_{0}_div_bid_1'.format(i)] = np.log(X['volume_bid_{0}'.format(i)] / X['volume_bid_1'])
        
    out_X['log_total_volume_ask'] = np.log(X[['volume_ask_{0}'.format(x) for x in list(range(1, level + 1))]].sum(axis = 1))
    out_X['log_total_volume_bid'] = np.log(X[['volume_bid_{0}'.format(x) for x in list(range(1, level + 1))]].sum(axis = 1))
            
    mid_price = (X['ask_1'] + X['bid_1']) / 2
    out_X['log_return_mid_price'] = np.log(mid_price.pct_change() + 1).shift(-1)
   
    cols_features = out_X.columns.drop(target_column)

    out_X = out_X.assign(**{
        '{}_(t-{})'.format(col, t): out_X[col].shift(t)
        for t in list(range(1, num_lags))
        for col in cols_features})

    return out_X.dropna()


def preprocess(case):
	return{
    		1:norm,
		2:minmax,
		3:max,
		4:log_return,
	
	}.get(case,f_default)

def calculate_accuracy(Y_true, Y_pred,result_high,result_low):
  predicted = Y_pred.ge(.5).view(-1) ##threshold 0.5
  return (Y_true == (predicted*(result_high-result_low)+result_low)).sum().float() / len(Y_true)

def round_tensor(t, decimal_places=5):
  return round(t.item(), decimal_places)
