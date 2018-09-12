#Tensorflow Movie Review Classification Model 
#2 Layer Neural Network with L1 Weight Regulariztion 
#Taweh Beysolow II 

import os, math, numpy as np, pandas as pan, tensorflow as tf 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#Paramters
tf.set_random_seed(2018)
learning_rate = 1e-4; epochs = 100; n_classes = 2
n_hidden = 1000; batch_size = 32

def summary_statistics(array):
    Min = min(array); Max = max(array); Range = Max - Min 
    Mean = np.mean(array); Sdev = np.std(array)
    output = pan.DataFrame([Min, Max, Range, Mean, Sdev]).T
    output.columns = ['Mean', 'Max', 'Range', 'Mean', 'SDev']
    return output
    
def remove_non_ascii(text):
    return ''.join([word for word in text if ord(word) < 128])
            
def load_data():
    negative_review_strings = os.listdir('/Users/tawehbeysolow/Downloads/review_data/tokens/neg')
    positive_review_strings = os.listdir('/Users/tawehbeysolow/Downloads/review_data/tokens/pos')
    negative_reviews, positive_reviews = [], []
    
    for positive_review in positive_review_strings:
        with open('/Users/tawehbeysolow/Downloads/review_data/tokens/pos/'+str(positive_review), 'r') as positive_file:
            positive_reviews.append(remove_non_ascii(positive_file.read()))
    
    for negative_review in negative_review_strings:
        with open('/Users/tawehbeysolow/Downloads/review_data/tokens/neg/'+str(negative_review), 'r') as negative_file:
            negative_reviews.append(remove_non_ascii(negative_file.read()))
    
    negative_labels, positive_labels = np.repeat(0, len(negative_reviews)), np.repeat(1, len(positive_reviews))
    labels = np.concatenate([negative_labels, positive_labels])
    reviews = np.concatenate([negative_reviews, positive_reviews])
    rows = np.random.random_integers(0, len(reviews)-1, len(reviews)-1)
    return reviews[rows], labels[rows]

def train_mlp(regularization, epochs=epochs, learning_rate=learning_rate):
    x, y = load_data(); print(x.shape)
    t = TfidfVectorizer(min_df=10, max_df=300, stop_words='english', token_pattern=r'\w+')
    x = t.fit_transform(x).todense(); train_end = int(math.floor(len(x)*.80))
    train_x, train_y = x[0:train_end] , np.array(pan.get_dummies(y[0:train_end]), dtype=int)
    test_x, test_y = x[train_end:] , y[train_end:]

    #Defining tensorflow placeholders and variables 
    X = tf.placeholder(tf.float32, shape=(None, train_x.shape[1]))
    Y = tf.placeholder(tf.int32, shape=(None, n_classes))

    weights = {'input': tf.Variable(tf.random_normal([train_x.shape[1], n_hidden])),
               'hidden': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
                'output': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
              
    biases = {'input': tf.Variable(tf.random_normal([n_hidden])),
              'hidden': tf.Variable(tf.random_normal([n_hidden])),
                'output': tf.Variable(tf.random_normal([n_classes]))}
    
    #Defining tensorflow graph and assorted graph operations 
    input_layer = tf.add(tf.matmul(X, weights['input']), biases['input'])
    input_layer = tf.nn.sigmoid(input_layer)
    hidden_layer = tf.add(tf.matmul(input_layer, weights['hidden']), biases['hidden'])
    hidden_layer = tf.nn.selu(hidden_layer)
    output_layer = tf.add(tf.matmul(hidden_layer, weights['output']), biases['output'])
    
    #Evaluating and backpropagating errors through neural network
    predictions = tf.nn.softmax(output_layer)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1)), tf.float32))
    cross_entropy = tf.reduce_mean(tf.cast(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=Y), tf.float32))
    if regularization == True:
        regularization = tf.contrib.layers.l2_regularizer(scale=0.0005, scope=None)
        regularization_penalty = tf.contrib.layers.apply_regularization(regularization, weights.values())
        cross_entropy = cross_entropy + regularization_penalty
    adam_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    with tf.Session() as sess:       
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs): #Cross-validating data
            rows = np.random.random_integers(0, len(train_x)-1, len(train_x)-1)
            _train_x, _train_y = train_x[rows], train_y[rows]
            
            #Batch training
            for start, end in zip(range(0, len(train_x)-1, batch_size), 
                                  range(batch_size, len(train_x)-1, batch_size)):
                __train_x, __train_y = _train_x[start:end], _train_y[start:end]
                _cross_entropy, _accuracy, _adam_optimizer = sess.run([cross_entropy, accuracy, adam_optimizer],
                                                         feed_dict={X:__train_x, Y:__train_y})
                
            if epoch%10 == 0 and epoch > 0:
                print('Epoch: ' + str(epoch) + 
                        '\nError: ' + str(_cross_entropy) +
                        '\nAccuracy: ' + str(_accuracy) + '\n')
        
        #Evaluating test model results
        predicted_y_values = pan.DataFrame(sess.run(predictions, feed_dict={X: test_x})).idxmax(axis=1)
        print('Test Set Accuracy Score: ' + str(accuracy_score(test_y, predicted_y_values)))
        print('Test Set Confusion Matrix: \n' + str(confusion_matrix(test_y, predicted_y_values)))
        
        false_positive_rate, true_positive_rate = roc_curve(test_y, predicted_y_values)[0], roc_curve(test_y, predicted_y_values)[1]
        auc_score = auc(false_positive_rate, true_positive_rate)
        plt.figure()
        plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' %auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for\n  MLP Model')
        plt.legend(loc="lower right")
        plt.show()
        
if __name__ == '__main__': 
    
    train_mlp(regularization=False)
