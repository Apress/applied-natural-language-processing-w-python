#Word Embeddings 
#Taweh Beysolow II 

#Import the necessary modules 
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf, numpy as np, string, math

#Parameters 
np.random.seed(2018)
epochs = 200
batch_size = 32
skip_gram_window_size = 2
cbow_window_size = 8
learning_rate = 1e-4
embedding_dim = 300
stop_words = stopwords.words('english')
punctuation = set(string.punctuation)
max_pages = 20
pdf_file = 'economics_textbook.pdf'

def euclidean_norm(vector):
    return np.sqrt(np.sum([_vector**2 for _vector in vector]))
       
def cosine_similarity(v1, v2):
    return np.dot(v1, v2)/float(euclidean_norm(v1)*euclidean_norm(v2))

def remove_non_ascii(text):
    return ''.join([word for word in text if ord(word) < 128])

def load_data(raw_text=False, pdf_file=pdf_file, max_pages=max_pages, directory='/Users/tawehbeysolow/Desktop/applied_nlp_python/data_etc/'):
    return_string = StringIO()
    device = TextConverter(PDFResourceManager(), return_string, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(PDFResourceManager(), device=device)
    filepath = file(directory+pdf_file, 'rb')
    for page in PDFPage.get_pages(filepath, set(), maxpages=max_pages, caching=True, check_extractable=True):
        interpreter.process_page(page)
    text_data = return_string.getvalue()
    filepath.close(), device.close(), return_string.close()
    if raw_text == True: return remove_non_ascii(text_data)
    else: text_data = ' '.join([word for word in word_tokenize(remove_non_ascii(text_data)) if word not in stop_words])
    return text_data
    
def gensim_preprocess_data(max_pages):
    data = load_data(max_pages=max_pages)
    sentences = sent_tokenize(data)
    tokenized_sentences = list([word_tokenize(sentence) for sentence in sentences])
    for i in range(0, len(tokenized_sentences)):
        tokenized_sentences[i] = [word for word in tokenized_sentences[i] if word not in punctuation]
    return tokenized_sentences
    
def gensim_skip_gram(max_pages=max_pages):
    sentences = gensim_preprocess_data(max_pages=max_pages)
    skip_gram = Word2Vec(sentences=sentences, window=1, min_count=10, sg=1)
    word_embedding = skip_gram[skip_gram.wv.vocab]
    pca = PCA(n_components=2)
    _word_embedding = pca.fit_transform(word_embedding)
    
    #Plotting results from trained word embedding
    plt.scatter(word_embedding[:, 0], word_embedding[:, 1])
    word_list = list(skip_gram.wv.vocab)
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(_word_embedding[i, 0], _word_embedding[i, 1])) 
        
    #Printing Cosine Similaritys of a few words
    for i in range(1, len(word_list)- 1):
        print(str('Cosine distance for %s  and %s' + 
              '\n ' + 
              str(cosine_similarity(word_embedding[i, :], word_embedding[i-1, :])))%(word_list[i], word_list[i-1]))

def tf_preprocess_data(window_size, skip_gram, max_pages):
        
    def one_hot_encoder(indices, vocab_size, skip_gram):
        vector = np.zeros(vocab_size)
        if skip_gram == True: vector[indices] = 1
        else:
            for index in indices: vector[index] = 1  
        return vector
        
    text_data = load_data(max_pages=max_pages)
    vocab_size, word_dictionary, index_dictionary, n_gram_data = len(word_tokenize(text_data)), {}, {},  []

    for index, word in enumerate(word_tokenize(text_data)):
        word_dictionary[word], index_dictionary[index] = index, word
           
    sentences = sent_tokenize(text_data) #Tokenizing sentences
    tokenized_sentences = list([word_tokenize(sentence) for sentence in sentences]) #Creating lists of words for each tokenized setnece

    for sentence in tokenized_sentences: #Creating word pairs for skip_gram model
        for index, word in enumerate(sentence):
            if word not in punctuation: #Removing grammatical objects from input data
                for _word in sentence[max(index - window_size, 0): min(index + window_size, len(sentence)) + 1]:
                    if _word != word: #Making sure not to duplicate word_1 when creating n-gram lists
                        n_gram_data.append([word, _word])
    
    x, y = np.zeros([len(n_gram_data), vocab_size]), np.zeros([len(n_gram_data), vocab_size])
    
    for i in range(0, len(n_gram_data)): #Concatenating one-hot encoded vector into input and output matrices
        x[i, :] = one_hot_encoder(word_dictionary[n_gram_data[i][0]], vocab_size=vocab_size, skip_gram=skip_gram)      
        y[i, :] = one_hot_encoder(word_dictionary[n_gram_data[i][1]], vocab_size=vocab_size, skip_gram=skip_gram)            

    return x, y, vocab_size, word_dictionary, index_dictionary

def tf_skip_gram_1(max_pages=max_pages, learning_rate=learning_rate, embedding_dim=embedding_dim):
    
    x, y, vocab_size, word_dictionary, index_dictionary = tf_preprocess_data(window_size=skip_gram_window_size, 
                                                                             skip_gram=True, max_pages=max_pages)
    
    #Defining tensorflow variables and placeholder
    X = tf.placeholder(tf.float32, shape=(None, vocab_size))
    Y = tf.placeholder(tf.float32, shape=(None, vocab_size))
    
    weights = {'hidden': tf.Variable(tf.random_normal([vocab_size, embedding_dim])),
               'output': tf.Variable(tf.random_normal([embedding_dim, vocab_size]))}

    biases = {'hidden': tf.Variable(tf.random_normal([embedding_dim])),
              'output': tf.Variable(tf.random_normal([vocab_size]))}
              
    input_layer = tf.add(tf.matmul(X, weights['hidden']), biases['hidden'])
    output_layer = tf.add(tf.matmul(input_layer, weights['output']), biases['output'])
    
    #Defining error, optimizer, and other objects to be used during training 
    cross_entropy = tf.reduce_mean(tf.cast(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=Y), tf.float32))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    
    #Executing graph 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):          
            rows = np.random.randint(0, int(math.floor(len(x)*.50)), int(math.floor(len(x)*.50)))
            _train_x, _train_y = x[rows], y[rows]

            #Batch training
            for start, end in zip(range(0, len(_train_x), batch_size), 
                                  range(batch_size, len(_train_x), batch_size)):
                
                _cross_entropy, _optimizer = sess.run([cross_entropy, optimizer], 
                                                      feed_dict={X:_train_x[start:end], Y: _train_y[start:end]})
                
            if epoch%10==0 and epoch > 1:
                print('Epoch: ' + str(epoch) + 
                        '\nError: ' + str(_cross_entropy) + '\n')
        
        word_embedding = sess.run(tf.add(weights['hidden'], biases['hidden']))
        pca = PCA(n_components=2)
        _word_embedding = pca.fit_transform(word_embedding)
        
        #Plotting results from trained word embedding
        plt.scatter(_word_embedding[0:50, 0], _word_embedding[0:50, 1])
        word_list = word_dictionary.keys()[0:50]
        for i, word in enumerate(word_list):
            plt.annotate(word, xy=(_word_embedding[i, 0], _word_embedding[i, 1]))

        #Printing Cosine Similaritys of a few words
        for i in range(1, len(word_list)- 1):
            print(str('Cosine distance for %s  and %s' + 
                  '\n ' + 
                  str(cosine_similarity(word_embedding[i, :], word_embedding[i-1, :])))%(word_list[i], word_list[i-1]))

       
def tf_skip_gram_2():
    
    x, y, vocab_size, word_dictionary, index_dictionary = tf_preprocess_data(window_size=skip_gram_window_size, skip_gram=True)
  
            
def gensim_cbow(max_pages):
    sentences = gensim_preprocess_data(max_pages=max_pages)
    
    cbow = Word2Vec(sentences=sentences, 
                    window=1, 
                    min_count=10, 
                    sg=0,
                    cbow_mean=0)
    
    word_embedding = cbow[cbow.wv.vocab]
    pca = PCA(n_components=2)
    word_embedding = pca.fit_transform(word_embedding)
    
    #Plotting results from trained word embedding
    plt.scatter(word_embedding[120:150, 0], word_embedding[120:150, 1])
    word_list = list(cbow.wv.vocab)
    for i in range(120, 150):
        plt.annotate(word_list[i], xy=(word_embedding[i, 0], word_embedding[i, 1]))
        
if __name__ == '__main__':
    
    #gensim_skip_gram()
    tf_skip_gram_1()
    #tf_skip_gram2()
    #tensorflow_cbow()
    #gensim_cbow(max_pages=100)
