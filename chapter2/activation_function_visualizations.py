#Visualization of Activation Functions 
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np
import matplotlib.pyplot as plt 

def plot_sigmoid():
    
    def sigmoid(x):
        return (1/float(1+np.exp(-x)))
    
    def deriv_sigmoid(x):
        return (sigmoid(x)* (1 - sigmoid(x)))
        
    sigmoid_array, deriv_array = [], []
    x = np.r_[-6:6, 6]
    #labels = np.array(x, dtype=str)
    for value in x:
        sigmoid_array.append(sigmoid(value))
        deriv_array.append(deriv_sigmoid(value))
       
    sigmoid, = plt.plot(sigmoid_array, label=['f(x)'])
    deriv, = plt.plot(deriv_array, label=['df/dx'])
    plt.title('Sigmoid Activation Function \n Derivative of Sigmoid Activation Function')
    plt.xlabel('X Value')
    plt.ylabel('Activation Function Output')
    plt.xticks(range(0, len(x)), x)
    plt.legend(handles=[sigmoid, deriv])
    

def plot_relu():
    
    def relu(x):
        return max(0, x)
        
    def deriv_relu(x):
        if x < 0: return 0
        elif x > 0: return 1
        else: return 0.5 
    
    relu_array, deriv_array = [], []
    x = np.r_[-6:6, 6]
    for value in x:
        relu_array.append(relu(value))
        deriv_array.append(deriv_relu(value))
    
    relu, = plt.plot(relu_array, label=['relu'])
    deriv, = plt.plot(deriv_array, label=['df/dx'])
    plt.title('ReLU Activation Function and \n Approximation of Derivative Function')
    plt.xlabel('X Value')
    plt.ylabel('Activation Function Output')
    plt.xticks(range(0, len(x)), x)
    plt.legend(handles=[relu, deriv])


def plot_tanh():
          
    def tanh(x):
        return ((np.exp(x) - np.exp(-x))/float(np.exp(x) + np.exp(-x)))
    
    def deriv_tanh(x):
        return (1 - tanh(x)**2)
        
    tanh_array, deriv_array = [], []
    x = np.r_[-6:6, 6]
    for value in x:
        tanh_array.append(tanh(value))
        deriv_array.append(deriv_tanh(value))
        
    tanh, = plt.plot(tanh_array, label=['f(x)'])
    deriv, = plt.plot(deriv_array, label=['df/dx'])
    plt.title('Tanh Activation Function and \n Derivative of Tanh Activation Function')
    plt.xlabel('X Value')
    plt.ylabel('Activation Function Output')
    plt.xticks(range(0, len(x)), x)
    plt.legend(handles=[tanh, deriv])
    
if __name__ == '__main__':

    #plot_sigmoid()
    plot_relu()
    #plot_tanh()