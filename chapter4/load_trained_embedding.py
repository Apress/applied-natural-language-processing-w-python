#Example of loading a trained word embedding 
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np, tensorflow as tf, matplotlib.pyplot as plt, collections, random
from chapter4.word_embeddings import load_data, cosine_similarity
from sklearn.decomposition import PCA
from tensorflow.contrib import rnn
from scipy import spatial

#Parameters
learning_rate = 1e-4; n_input = 4; 
n_hidden = 500; epochs = 3000 
offset=10; n_units = n_hidden


sample_text = '''Living in different places has been the greatest experience 
that I have had in my life. It has allowed me to understand people from 
different walks of life, as well as to question some of my own biases I have had 
with respect to people who did not grow up as I did. If possible, everyone should 
take an opportunity to travel somewhere separate from where they grew up'''.replace('\n', '')

#sample_text = load_data().replace('\n', '')

def load_embedding(embedding_path='/Users/tawehbeysolow/Downloads/glove.6B.50D.txt'):
    vocabulary, embedding, embedding_dictionary = [], [], {}
    for line in open(embedding_path, 'rb').readlines():
        row = line.strip().split(' ')
        vocabulary.append(row[0]), embedding.append(row[1:])
        embedding_vector = [float(i) for i in row[1:]]
        embedding_dictionary[row[0]] = embedding_vector
    vocabulary_length, embedding_dim = len(vocabulary), len(embedding[0])
    return vocabulary, np.asarray(embedding, dtype=float), vocabulary_length, embedding_dim, embedding_dictionary

def visualize_embedding_example():
    
    vocabulary, embedding, vocabulary_length, embedding_dim, embedding_dictionary = load_embedding()

    #Showing example of pretrained word embedding vectors
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(embedding)
    plt.scatter(pca_embedding[0:50, 0], pca_embedding[0:50, 1])
    for i, word in enumerate(vocabulary[0:50]):
        plt.annotate(word, xy=(pca_embedding[i, 0], pca_embedding[i, 1]))
        
    #Comparing cosine similarity 
    for k in range(100, 150):
        text = str('Cosine Similarity Between %s and %s: %s')%(vocabulary[k],
                                                            vocabulary[k-1], 
                                                cosine_similarity(embedding[k], 
                                                                  embedding[k-1]))
        print(text)

        
def training_data_example(sample_text=sample_text, learning_rate=learning_rate, 
                          n_input=n_input, n_hidden=n_hidden, epochs=epochs,
                          offset=offset, n_units=n_units):
    
    vocabulary, embedding, vocabulary_length, embedding_dim, embedding_dictionary = load_embedding()
    _sample_text = np.array(sample_text.split())
    _sample_text = _sample_text.reshape([-1, ])
    _embedding_array = []

    def sample_text_dictionary(data=_sample_text):
        count, dictionary = collections.Counter(data).most_common(), {} #creates list of word/count pairs;
        for word, _ in count:
            dictionary[word] = len(dictionary) #len(dictionary) increases each iteration
            reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        dictionary_list = sorted(dictionary.items(), key = lambda x : x[1])
        return dictionary, reverse_dictionary, dictionary_list
        
    dictionary, reverse_dictionary, dictionary_list = sample_text_dictionary()
    
    for i in range(len(dictionary)):
        word = dictionary_list[i][0]
        if word in vocabulary:
            _embedding_array.append(embedding_dictionary[word])
        else:
            _embedding_array.append(np.random.uniform(low=-0.2, high=0.2, size=embedding_dim))
     
    embedding_array = np.asarray(_embedding_array)     
    decision_tree = spatial.KDTree(embedding_array, leafsize=100)

    #Initializing placeholders and other variables
    X = tf.placeholder(tf.int32, shape=(None, None, n_input))
    Y = tf.placeholder(tf.float32, shape=(None, embedding_dim))
    weights = {'output': tf.Variable(tf.random_normal([n_hidden, embedding_dim]))}
    biases = {'output': tf.Variable(tf.random_normal([embedding_dim]))}
    
    _weights = tf.Variable(tf.constant(0.0, shape=[vocabulary_length, embedding_dim]), trainable=True)
    _embedding = tf.placeholder(tf.float32, [vocabulary_length, embedding_dim])
    embedding_initializer = _weights.assign(_embedding)
    embedding_characters = tf.nn.embedding_lookup(_weights, X)
        
    input_series = tf.reshape(embedding_characters, [-1, n_input])
    input_series = tf.split(input_series, n_input, 1)
    lstm_cell =  rnn.BasicLSTMCell(num_units=n_units, state_is_tuple=True, reuse=None, activation=tf.nn.relu)
    outputs, states = rnn.static_rnn(lstm_cell, input_series, dtype=tf.float32)
    
    output_layer = tf.add(tf.matmul(outputs[-1], weights['output']), biases['output'])
    error = tf.reduce_mean(tf.nn.l2_loss(output_layer - Y))
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            
            if offset+n_input >= len(reverse_dictionary): 
                offset = random.randint(0, n_input+1)
           
            sess.run(embedding_initializer, feed_dict={_embedding: embedding})
            
            #Creatin input and output training data
            x_train = [[dictionary[str(_sample_text[i])]] for i in range(offset, offset+n_input)]
            x_train = np.reshape(np.array(x_train), [-1, 1, n_input])
            y_train = dictionary[_sample_text[offset+n_input]]
            y_train = embedding[y_train, :]
            y_train = np.reshape(y_train, [1, -1])
    
            _, _error, _prediction = sess.run([adam_optimizer, error, output_layer], 
                                     feed_dict = {X: x_train, Y: y_train})
            
            if epoch%10 == 0 and epoch > 0:
                input_sequence = [str(_sample_text[i]) for i in range(offset, offset+n_input)] 
                target_word = str(_sample_text[offset+n_input])
                distance, _index = decision_tree.query(_prediction[0], 1)
                predicted_word = reverse_dictionary[_index]
                  
                print('Input Sequence: %s \nActual Label: %s \nPredicted Label: %s'%
                      (input_sequence, target_word, predicted_word))
                print('Epoch: %s \nError: %s \n'%(epoch, _error))
                offset += (n_input+1) 
                

if __name__ == '__main__':

    #visualize_embedding_example()
    
    training_data_example()
