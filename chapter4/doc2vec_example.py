#Doc2Vec Example 
#Taweh Beysolow II 

#Import the necessary modules 
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Doc2Vec
from collections import namedtuple 
from chapter4.word_embeddings import load_data, cosine_similarity
import time

#Parameters
stop_words = stopwords.words('english')
learning_rate = 1e-4
epochs = 200
max_pages = 300

sample_text1 = '''I love italian food. My favorite items are pizza and pasta,
especially garlic bread. The best italian food I have had has been in New York. 
Little Italy was very fun.'''

sample_text2 = '''My favorite time of italian food is pasta with alfredo sauce. 
It is very creamy but the cheese is the best part. Whenevr I go to an italian 
restaurant, I am always certain to get a plate.'''

def gensim_preprocess_data(max_pages):
    sentences = namedtuple('sentence', 'words tags')
    _sentences = sent_tokenize(load_data(max_pages=max_pages))
    documents = []
    for i, text in enumerate(_sentences):
        words, tags = text.lower().split(), [i]
        documents.append(sentences(words, tags))
    return documents

def train_model(training_example, max_pages=max_pages, epochs=epochs, learning_rate=learning_rate):
    sentences = gensim_preprocess_data(max_pages=max_pages)
    model = Doc2Vec(alpha=learning_rate, min_alpha=learning_rate/float(3))
    model.build_vocab(sentences)
    model.train(documents=sentences, total_examples=len(sentences), epochs=epochs)
    
    #Showing distance between different documents 
    if training_example == True:
        
        for i in range(1, len(sentences)-1):
            print(str('Document ' + str(sentences[i-1]) + '\n'))
            print(str('Document ' + str(sentences[i]) + '\n'))
            print('Cosine Similarity Between Documents: ' + 
                  '\n' + str(cosine_similarity(model.docvecs[i-1], model.docvecs[i])))
            
            time.sleep(10)
                    
    else:
   
        print('Cosine Similarity Between Sample Texts: ' + 
          '\n' + str(cosine_similarity(model.infer_vector(sample_text1.lower().split()), 
                                        model.infer_vector(sample_text2.lower().split()))))

if __name__ == '__main__': 
    
    #train_model(training_example=True)
    train_model(training_example=False)