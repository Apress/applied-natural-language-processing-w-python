#GloVe Word Embedding Implementation 
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np, matplotlib.pyplot as plt, string, itertools, tensorflow as tf
from chapter4.word_embeddings import load_data, cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Parameters
stop_words = stopwords.words('english')
punctuation = set(string.punctuation)

def weighting_function(x, x_max=100, alpha=3/float(4)):
    return (x/float(x_max))**alpha if x < x_max else 1

def plotting_weighting_function():
    input_array = np.r_[0:200, 200]
    output_array = []
    for value in input_array: output_array.append(weighting_function(x=value))
    
    #Plotting function 
    plt.plot(output_array, 'green')
    plt.xlabel('x value')
    plt.ylabel('f(x)')
    plt.title('Weighting Function Output over Input Value')
    return None
    
def preprocess_data(max_pages):
    raw_data = word_tokenize(load_data(max_pages=max_pages))
    vocabulary = np.unique(raw_data)
    vocabulary = [word for word in vocabulary if word not in stop_words and word not in punctuation and word != "''"]
    raw_data = [word for word in raw_data if word not in stop_words and word not in punctuation and word != "''"]
    counts, term_frequency, vocabulary_size = np.zeros([1, len(vocabulary)]), {}, len(vocabulary)

    for k in range(0, vocabulary_size): #Obtaining term-frequencies and transforming to dictionary
        print(str(k/float(vocabulary_size)) + ' percent complete with term-frequency dictionary generation...')
        for i in range(0, len(raw_data)):
            if vocabulary[k] == raw_data[i]: 
                counts[0][k] += 1
                term_frequency[vocabulary[k]] = (k, counts[0][k])

    return term_frequency, raw_data
    
def cooccurrence_matrix(max_pages=30):
    term_frequency, raw_data = preprocess_data(max_pages=max_pages)
    word_id_dictionary = {}

    for i in range(0, len(term_frequency.values())):
        word_id_dictionary[term_frequency.keys()[i]] = i
                           
    cooccurrence_matrix = np.zeros((len(term_frequency.values()), len(term_frequency.values())))
    for k in range(0, len(word_id_dictionary)):
        print(str(k/float(len(word_id_dictionary))) + ' percent complete with co-occurence matrix generation...')
        for i in range(1, len(raw_data)-1): 
            if raw_data[i-1] == word_id_dictionary.keys()[k]:
                cooccurrence_matrix[word_id_dictionary.values()[k], word_id_dictionary[raw_data[i]]] += 1

    #Converting co-occurences to probabilities 
    for k in range(0, cooccurence_matrix.shape[1]):
        for i in range(0, len(cooccurence_matrix)):
            cooccurence_matrix[i, k] = cooccurence_matrix[i, k]/float(cooccurrence_matrix[i ,:].sum())
            
    #Creating input data for matrix 
    
    
    
    
    
    
    
    return cooccurrence_matrix, len(term_frequency)

def glove_implementation(embedding_dim):
    
    input_data, vocabulary_size = cooccurrence_matrix()
     
    X = tf.placeholder(tf.float32, shape=(None, embedding_dim))
    Y = tf.placeholder(tf.float, shape=(None, vocabulary_size))
    
    weights = {'input': tf.Variable(tf.random_normal([vocabulary_size, embedding_dim])),
               'output': tf.Varibale(tf.random_normal([embedding_dim, vocabulary_size]))}

    biases = {'input': tf.Variable(tf.random_normal([embedding_dim])),
              'output': tf.Variable(tf.random_normal([vocabulary_size]))}


    #Defining tensorflow operations for GloVe
    input_layer = tf.add(tf.matmul(X, weights['input']), biases['input'])
    output_layer = tf.add(tf.matmul(X, weights['output']), biases['output'])

    
if __name__ == '__main__':
    
    cooccurrence_matrix()