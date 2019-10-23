import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential
from keras.layers import *
from pylab import *
from sklearn.preprocessing import StandardScaler
from array import *
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#import model_to_estimator 
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  

#Training Data
ALL_DATA= pd.read_csv('../TSLA.csv')
PRICE_DATA = ALL_DATA['Close'].values
DATE_DATA=ALL_DATA['Date'].values
print('Number of days I am looking at: '+ str((len(PRICE_DATA))))

#x=np.array(xarr)
#y=np.array(yarr)


#SCALING DATA
scaler = StandardScaler()
scaled_price = scaler.fit_transform(PRICE_DATA.reshape(-1, 1))

#PLOTTING RAW DATA
plt.figure(figsize=(12,7), frameon=False, facecolor='brown', edgecolor='blue')
plt.title('Scaled TESLA Stock Price from August 2015 to August 2019')
plt.xlabel('Days')
plt.ylabel('Price of Stocks')
plt.plot(PRICE_DATA, label='Stocks data')
plt.legend()
plt.show()

#SPLITS THE DATA INTO 2
   #TRAINING DATA

x_train  = np.array(PRICE_DATA[0:int(len(PRICE_DATA)*0.7)])
y_train = np.array(DATE_DATA[0:int(len(PRICE_DATA)*0.7)])
   #TEST DATA
x_test = np.array(PRICE_DATA[int(len(PRICE_DATA)*0.7): len(PRICE_DATA)])
y_test = np.array(DATE_DATA[int(len(PRICE_DATA)*0.7): len(PRICE_DATA)])

#RESHAPE DATA SO ITS 3D FOR LTSM
x_train=x_train.reshape(1,len(x_train),1)
y_train=y_train.reshape(1,len(y_train),1)

'''
# reshape 1D array
from numpy import array
from numpy import reshape
# define array
data = array([11, 22, 33, 44, 55])
print(data.shape)
# reshape
data = data.reshape((data.shape[0], 1))
print(data.shape)
1
2
3
4
5
6
7
8
9
# reshape 1D array
from numpy import array
from numpy import reshape
# define array
data = array([11, 22, 33, 44, 55])
print(data.shape)
# reshape
data = data.reshape((data.shape[0], 1))
print(data.shape)

# reshape 2D array
from numpy import array
# list of data
data = [[11, 22],
		[33, 44],
		[55, 66]]
# array of data
data = array(data)
print(data.shape)
# reshape
data = data.reshape((data.shape[0], data.shape[1], 1))
print(data.shape)

'''

#BUILD MODEL/TRAIN IT
model = Sequential()

#model.add(LSTM(64, input_shape=(1,4)))
#model.add(Dense(1))
model.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (880, 1)))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
model.fit(x_train, y_train, batch_size = 32, epochs = 200)

predicted_stock_price = regressor.predict(x_test)


#take test data set, and use baseline and actual model and evaluate on the test data

#model.compile(loss="mean_squared_error", optimizer="sgd")

#model.fit(x_train,y_train,epochs=1000,verbose=0, batch_size=1, shuffle=False)

#TEST MODEL
#mse = model.evaluate(x_test,y_test, verbose=0)
#print('Mean Squared Error: ', mse)

#PREDICTION
#prediction=model.predict(x_test)
#prediction=predict(x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
Predict = model.predict(x_test)
testScore = math.sqrt(mean_squared_error(y_test, Predict))
print('Test Score: %.2f RMSE' % (testScore))
plt.plot(y_test)
plt.plot(Predict)
plt.legend(['original value','predicted value'],loc='upper right')
plt.show()




'''
#GRAPH REGRESSION
plt.figure()
plt.plot((X1-s.min_[0])/s.scale_[0], \
                 (Y1-s.min_[1])/s.scale_[1], \
                 'bo',label='train')
plt.plot(x,y,'ro',label='actual')
plt.plot(x,yp,'k--',label='predict')
plt.legend(loc='best')
plt.savefig('results.png')
plt.show()
'''



'''
# Graph of Loss as epochs increase

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

tf_classifier = tf.keras.estimator.model_to_estimator(keras_model=model)
'''