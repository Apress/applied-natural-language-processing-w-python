#RNN Tensorflow Demo
#Taweh Beysolow II

#Import necessary modules
import numpy as np, pandas as pan, tensorflow as tf, math 
from keras.utils import np_utils
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler

#Setting parameters to be used
learning_rate = 0.02; epochs = 600
series_len = 50000; state_size = 4; batch_size = 32
    
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
    
#Vanilla RNN Implementation   
def make_rnn(learning_rate=0.02, epochs=600, state_size=4, n_hidden=300):
    
    x, y = load_data(); scaler = MinMaxScaler(feature_range=(0, 1))
    x, y = scaler.fit_transform(x), scaler.fit_transform(y)
    train_x, train_y = x[0:int(math.floor(len(x)*.67)),  :], y[0:int(math.floor(len(y)*.67))]
    
    #Creating weights and biases dictionaries
    weights = {'input': tf.Variable(tf.random_normal([state_size+1, state_size])),
        'output': tf.Variable(tf.random_normal([state_size, train_y.shape[1]]))}
    biases = {'input': tf.Variable(tf.random_normal([1, state_size])),
        'output': tf.Variable(tf.random_normal([1, train_y.shape[1]]))}

    #Defining placeholders and variables
    X = tf.placeholder(tf.float32, [batch_size, train_x.shape[1]])
    Y = tf.placeholder(tf.int32, [batch_size, train_y.shape[1]])
    init_state = tf.placeholder(tf.float32, [batch_size, state_size])
    input_series = tf.unstack(X, axis=1)
    labels = tf.unstack(Y, axis=1)
    current_state = init_state
    hidden_states = []

    #Passing values from one hidden state to the next
    for input in input_series: #Evaluating each input within the series of inputs
        input = tf.reshape(input, [batch_size, 1]) #Reshaping input into MxN tensor
        input_state = tf.concat([input, current_state], axis=1) #Concatenating input and current state tensors
        _hidden_state = tf.tanh(tf.add(tf.matmul(input_state, weights['input']), biases['input'])) #Tanh transformation
        hidden_states.append(_hidden_state) #Appending the next state
        current_state = _hidden_state #Updating the current state

    logits = [tf.add(tf.matmul(state, weights['output']), biases['output']) for state in hidden_states]
    predicted_labels = [tf.nn.softmax(logit) for logit in logits] #predictions for each logit within the series
    error = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label) for logit, label in zip(logits, labels)]
    cross_entropy = tf.reduce_mean(error)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(predicted_labels, 1)), tf.float32))

    #Execute Graph
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
  
            _state = np.zeros([batch_size, state_size])

            for start, end in zip(range(0, len(train_x)-batch_size, batch_size), range(batch_size, len(train_x), batch_size)):
                
                _train_x, _train_y = train_x[start:end, :], train_y[start:end]
                _error, _error, _state, _accuracy = sess.run([optimizer, cross_entropy, init_state, accuracy],
                                                                  feed_dict={X:_train_x, Y:_train_y, init_state:_state})
                
            if epoch%20 == 0:
                print('Epoch: ' + str(epoch) +  
                '\nError:' + str(_error) + 
                '\nAccuracy: ' + str(_accuracy) + '\n')

    
if __name__ == '__main__': 
    
    make_rnn()
