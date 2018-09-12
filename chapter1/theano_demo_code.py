#Demo Theano Code
import theano
from theano import tensor as T
import numpy as np
from mnist import MNIST

#Functions and parameters to be used later
mndata = MNIST('/Users/tawehbeysolow/Downloads')
weight_shape = (784, 10)
bias_shape = (1, 10)
batch_size = 128

def float_x(X):
    return np.asarray(X, dtype=theano.config.floatX)
    
def float_y(y):
    return np.asarray(y, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(float_x(np.random.randn(*shape) * 0.01))
    
def init_biases(shape):
    return theano.shared(float_y(np.random.randn(*shape) * 0.01))

def model(X, w, b):
    return T.nnet.softmax(T.dot(X, w) + b)

def model_predict():
    train_x_data, train_y_data = mndata.load_training()
    test_x_data, test_y_data = mndata.load_testing()
    
    X, Y = T.fmatrix(), T.vector(dtype=theano.config.floatX)    
    weights = init_weights(weight_shape)
    biases = init_biases(bias_shape)
    predicted_y = T.argmax(model(X, weights, biases), axis=1)
    
    cost = T.mean(T.nnet.categorical_crossentropy(predicted_y, Y))
    gradient = T.grad(cost=cost, wrt=weights)
    update = [[weights, weights - gradient * 0.05]]
    
    train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=predicted_y, allow_input_downcast=True)
    
    for i in range(0, 10):
        print(predict(test_x_data[i:i+1]))

if __name__ == '__main__':
    
    model_predict()