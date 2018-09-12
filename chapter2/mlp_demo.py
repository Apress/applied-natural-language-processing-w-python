#Tensorflow Demos
#Taweh Beysolow II

import tensorflow as tf, numpy as np, pandas as pan
import pandas_datareader as data
tf.reset_default_graph()

def load_data():
    tickers = ["F", "SPY","DIA", "HAL", "MSFT", "SWN", "SJM", "SLG"]
    raw_data = pan.DataFrame()
    
    for i in range(0, len(tickers)):
        print(str((i/float(len(tickers)))*100) + ' percent complete with loading training data...')
        raw_data = pan.concat([raw_data, data.DataReader(tickers[i], data_source = 'yahoo', 
                                                             start = '2010-01-01', 
                                                             end = '2017-01-01')['Close']], axis=1)
        
    #Renaming Coliumns
    raw_data.columns = tickers
    
    #Calculating returns on stocks 
    stock_returns = np.matrix(np.zeros([raw_data.shape[0], raw_data.shape[1]]))
    for j in range(0, raw_data.shape[1]):
        for i in range(0, len(raw_data)-1):
            stock_returns[i,j] = raw_data.ix[i+1,j]/raw_data.ix[i,j] - 1
    
    return stock_returns

train_data = load_data()
print(pan.DataFrame(train_data[0:10, 1:]))
   
def mlp_model(train_data=train_data, learning_rate=0.01, num_hidden=256, epochs=100):
    
    #Creating training and test sets
    train_x, train_y = train_data[0:int(len(train_data)*.67), 1:train_data.shape[1]], train_data[0:int(len(train_data)*.67), 0]                     
    test_x, test_y = train_data[int(len(train_data)*.67):, 1:train_data.shape[1]], train_data[int(len(train_data)*.67):, 0]
    
    #Creating placeholder values and instantiating weights and biases as dictionaries
    X = tf.placeholder('float', shape = (None, 7))
    Y = tf.placeholder('float', shape = (None, 1))
    
    weights = {'input': tf.Variable(tf.random_normal([train_x.shape[1], num_hidden])),
            'hidden1': tf.Variable(tf.random_normal([num_hidden, num_hidden])),
            'output': tf.Variable(tf.random_normal([num_hidden, 1]))}

    biases = {'input': tf.Variable(tf.random_normal([num_hidden])),
            'hidden1': tf.Variable(tf.random_normal([num_hidden])),
            'output': tf.Variable(tf.random_normal([1]))}
    
    #Passing data through input, hidden, and output layers
    input_layer = tf.add(tf.matmul(X, weights['input']), biases['input'])
    input_layer = tf.nn.sigmoid(input_layer)
    input_layer = tf.nn.dropout(input_layer, 0.20)
    
    hidden_layer = tf.add(tf.multiply(input_layer, weights['hidden1']), biases['hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    hidden_layer = tf.nn.dropout(hidden_layer, 0.20)
    
    output_layer = tf.add(tf.multiply(hidden_layer, weights['output']), biases['output'])
    
    #Evaluating error and updating weights via gradient descent
    error = tf.reduce_sum(tf.pow(output_layer - Y, 2))/(len(train_x))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
    
    #Executing graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(epochs):
            for _train_x, _train_y in zip(train_x, train_y):
                _, _error = sess.run([optimizer, error], feed_dict={X:_train_x, Y:_train_y })
            
            #Printing Logging Information 
            print('Epoch ' +  str((i+1)) + ' Error: ' + str(_error))
                    
        #Predicting out of sample
        test_error = []
        for _test_x, _test_y, in zip(test_x, test_y):
            test_error.append(sess.run(error, feed_dict={X:_test_x, Y:_test_y}))
        print('Test Error: ' + str(np.sum(test_error)))
 
'''
DEPRECATED: NOT A VALID SOLUTION
def vanishing_gradient(train_data=train_data, activation=tf.nn.relu, learning_rate=0.01, num_hidden=256, epochs=100):
                       
    #Creating training and test sets
    train_x, train_y = train_data[0:int(len(train_data)*.67), 1:train_data.shape[1]], train_data[0:int(len(train_data)*.67), 0]                     
    test_x, test_y = train_data[int(len(train_data)*.67):, 1:train_data.shape[1]], train_data[int(len(train_data)*.67):, 0]
    
    #Creating placeholder values and instantiating weights and biases as dictionaries
    X = tf.placeholder('float', shape = (None, 7))
    Y = tf.placeholder('float', shape = (None, 1))
    
    weights = {'input': tf.Variable(tf.random_normal([train_x.shape[1], num_hidden])),
            'hidden1': tf.Variable(tf.random_normal([num_hidden, num_hidden])),
            'output': tf.Variable(tf.random_normal([num_hidden, 1]))}

    biases = {'input': tf.Variable(tf.random_normal([num_hidden])),
            'hidden1': tf.Variable(tf.random_normal([num_hidden])),
            'output': tf.Variable(tf.random_normal([1]))}
    
    #Passing data through input, hidden, and output layers
    input_layer = tf.add(tf.matmul(X, weights['input']), biases['input'])
    input_layer = activation(input_layer)
    #input_layer = tf.nn.dropout(input_layer, 0.20)
    
    hidden_layer = tf.add(tf.multiply(input_layer, weights['hidden1']), biases['hidden1'])
    hidden_layer = activation(hidden_layer)
    #hidden_layer = tf.nn.dropout(hidden_layer, 0.20)
    
    output_layer = tf.add(tf.multiply(hidden_layer, weights['output']), biases['output'])
    
    #Evaluating error and updating weights via gradient descent
    error = tf.reduce_sum(tf.pow(output_layer - Y, 2))/(len(train_x))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
    
    #Executing graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(epochs):
            for _train_x, _train_y in zip(train_x, train_y):
                sess.run(optimizer, feed_dict={X:_train_x, Y:_train_y })
            
            #Printing Logging Information 
            _error = sess.run(error, feed_dict={X:_train_x, Y:_train_y})
            print('Epoch ' +  str((i+1)) + ' Error: ' + str(_error))
        
        #Predicting out of sample
        test_error = []
        for _test_x, _test_y, in zip(test_x, test_y):
            test_error.append(sess.run(error, feed_dict={X:_test_x, Y:_test_y}))
        print('Test Error: ' + str(np.sum(test_error)))
'''
        
def mlp_no_dropout(train_data=train_data, learning_rate=0.01, num_hidden=256, epochs=100):
    
    #Creating training and test sets
    train_x, train_y = train_data[0:int(len(train_data)*.67), 1:train_data.shape[1]], train_data[0:int(len(train_data)*.67), 0]                     
    test_x, test_y = train_data[int(len(train_data)*.67):, 1:train_data.shape[1]], train_data[int(len(train_data)*.67):, 0]
    
    #Creating placeholder values and instantiating weights and biases as dictionaries
    X = tf.placeholder('float', shape = (None, 7))
    Y = tf.placeholder('float', shape = (None, 1))
    
    weights = {'input': tf.Variable(tf.random_normal([train_x.shape[1], num_hidden])),
            'hidden1': tf.Variable(tf.random_normal([num_hidden, num_hidden])),
            'output': tf.Variable(tf.random_normal([num_hidden, 1]))}

    biases = {'input': tf.Variable(tf.random_normal([num_hidden])),
            'hidden1': tf.Variable(tf.random_normal([num_hidden])),
            'output': tf.Variable(tf.random_normal([1]))}
    
    #Passing data through input, hidden, and output layers
    input_layer = tf.add(tf.matmul(X, weights['input']), biases['input'])
    input_layer = tf.nn.sigmoid(input_layer)
    
    hidden_layer = tf.add(tf.multiply(input_layer, weights['hidden1']), biases['hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    
    output_layer = tf.add(tf.multiply(hidden_layer, weights['output']), biases['output'])
    
    #Evaluating error and updating weights via gradient descent
    error = tf.reduce_sum(tf.pow(output_layer - Y, 2))/(len(train_x))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
    
    #Executing graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(epochs):
            for _train_x, _train_y in zip(train_x, train_y):
                _, _error = sess.run([optimizer, error], feed_dict={X:_train_x, Y:_train_y })
            
            #Printing Logging Information 
            print('Epoch ' +  str((i+1)) + ' Error: ' + str(_error))
        
        #Predicting out of sample
        test_error = []
        for _test_x, _test_y, in zip(test_x, test_y):
            test_error.append(sess.run(error, feed_dict={X:_test_x, Y:_test_y}))
        print('Test Error: ' + str(np.sum(test_error)))
                

#Making predictions with trained model
if __name__ == '__main__':
     
    #mlp_no_dropout()
    mlp_model() 
