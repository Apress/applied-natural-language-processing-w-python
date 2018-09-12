#LSTM Tensorflow Demo 
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np, pandas as pan, tensorflow as tf, math
from tensorflow.contrib import rnn
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils

#Parameters
input_shape=(1, 5); n_hidden=300; batch_size=200; learning_rate=1e-3; 
training_steps=1000; epochs=100; series_len=10000; state_size=4;
backprop_len = 15; tf.reset_default_graph()

def load_data():
    tickers = ["F", "SPY","DIA", "HAL", "MSFT", "SWN", "SJM", "SLG"]
    raw_data = pan.DataFrame()
    
    for i in range(0, len(tickers)):
        print(str((i/float(len(tickers)))*100) + ' percent complete with loading training data...')
        raw_data = pan.concat([raw_data, data.DataReader(tickers[i], data_source = 'yahoo', 
                                                             start = '2010-01-01', 
                                                             end = '2017-01-01')['Close']], axis=1)
        
    raw_data.columns = tickers
    stock_returns = np.matrix(np.zeros([raw_data.shape[0], raw_data.shape[1]]))
    for j in range(0, raw_data.shape[1]):
        for i in range(0, len(raw_data)-1):
            stock_returns[i,j] = raw_data.ix[i+1,j]/raw_data.ix[i,j] - 1
    y = [1 if stock_returns[i, 0] > stock_returns[i-1, 0] else  0 for i in range(1, len(stock_returns))]
    y = np_utils.to_categorical(y[1:])
    x = stock_returns[0:len(stock_returns)-2, :]                         
    return x, y

def train_lstm(learning_rate=learning_rate, n_units=n_hidden, epochs=epochs):

    x, y = load_data(); scaler = MinMaxScaler(feature_range=(0, 1))
    x, y = scaler.fit_transform(x), scaler.fit_transform(y)
    train_x, train_y = x[0:int(math.floor(len(x)*.67)),  :], y[0:int(math.floor(len(y)*.67))]
    
    X = tf.placeholder(tf.float32, (None, None, train_x.shape[1]))
    Y = tf.placeholder(tf.float32, (None, train_y.shape[1]))
    weights = {'output': tf.Variable(tf.random_normal([n_hidden, train_y.shape[1]]))}
    biases = {'output': tf.Variable(tf.random_normal([train_y.shape[1]]))}
    input_series = tf.reshape(X, [-1, train_x.shape[1]])
    input_series = tf.split(input_series, train_x.shape[1], 1)
    
    lstm = rnn.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0, reuse=None, state_is_tuple=True)
    _outputs, states = rnn.static_rnn(lstm, input_series, dtype=tf.float32)
    predictions = tf.add(tf.matmul(_outputs[-1], weights['output']), biases['output'])   
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(predictions), 1),
                                               tf.argmax(Y, 1)), dtype=tf.float32))
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=predictions))
    adam_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
          
            for start, end in zip(range(0, len(train_x)-batch_size, batch_size), 
                                  range(batch_size, len(train_x), batch_size)):
                
                _train_x, _train_y = train_x[start:end], train_y[start:end]
                _train_x = _train_x.reshape(_train_x.shape[0], 1, train_x.shape[1])
                _, _error, _accuracy = sess.run([adam_optimizer, error,  accuracy],  
                                                feed_dict={X: _train_x, Y: _train_y})
                
            if epoch%10 == 0:
                print('Epoch: ' + str(epoch) +  
                '\nError:' + str(_error) + 
                '\nAccuracy: ' + str(_accuracy) + '\n')
                
if __name__ == '__main__':
    
    train_lstm()