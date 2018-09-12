#Chapter 4 LDA Example
#Taweh Beysolow II 

#import the necessary modules 
import numpy as np 
import re, gensim#, pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from chapter3.movie_review_classification import load_data
from nltk import word_tokenize
from nltk.corpus import stopwords
 
np.random.seed(2018)
n_topics = 10
stop_words = stopwords.words('english')
n_frequent_words = 1500
n_components = 10

def print_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
        
def sklearn_topic_model():
    
    def create_topic_model(model, n_topics=10, max_iter=5, min_df=10, 
                           max_df=300, stop_words='english', token_pattern=r'\w+'):
        
        print(model + ' topic model:')
        data = load_data()[0] 
        if model == 'tf':
            feature_extractor = CountVectorizer(min_df=min_df, max_df=max_df, 
                                                stop_words=stop_words, token_pattern=token_pattern)
        else:
            feature_extractor = TfidfVectorizer(min_df=min_df, max_df=max_df, 
                                                stop_words=stop_words, token_pattern=token_pattern)
            
        processed_data = feature_extractor.fit_transform(data) 
        lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', 
                                              learning_offset=50., max_iter=max_iter, verbose=0)      
        lda_model.fit(processed_data)       
        tf_features = feature_extractor.get_feature_names()
        print_topics(model=lda_model, feature_names=tf_features, n_top_words=n_topics)
        #return lda_model, processed_data, feature_extractor

    create_topic_model(model='tf')        
  
def gensim_topic_model():
    
    def remove_stop_words(text):
        word_tokens = word_tokenize(text.lower())
        word_tokens = [word for word in word_tokens if word not in stop_words and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', word)]
        return word_tokens

    data = load_data()[0]
    cleaned_data = [remove_stop_words(data[i]) for i in range(0, len(data))]  
    dictionary = gensim.corpora.Dictionary(cleaned_data)
    dictionary.filter_extremes(no_below=500, no_above=1000)
    corpus = [dictionary.doc2bow(text) for text in cleaned_data]             
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary)   
    print('Gensim LDA implemenation: ')
    for _id in range(n_topics):
        header = str('Topic #%s: '%(_id))  
        tail = str(lda_model.print_topic(_id, 10))
        print(header + tail)
    
def nmf_topic_model():

    def create_topic_model(model='tf', n_topics=10, max_iter=5, min_df=10, 
                           max_df=300, stop_words='english', token_pattern=r'\w+'):
        print(model + ' NMF topic model: ')
        data = load_data()[0]
        if model == 'tf':
            feature_extractor = CountVectorizer(min_df=min_df, max_df=max_df, 
                                                stop_words=stop_words, token_pattern=token_pattern)
        else:
            feature_extractor = TfidfVectorizer(min_df=min_df, max_df=max_df, 
                                                stop_words=stop_words, token_pattern=token_pattern)

        processed_data = feature_extractor.fit_transform(data)
        nmf_model = NMF(n_components=n_components, max_iter=max_iter)      
        nmf_model.fit(processed_data)
        tf_features = feature_extractor.get_feature_names()
        print_topics(model=nmf_model, feature_names=tf_features, n_top_words=n_topics)
        return nmf_model, processed_data, feature_extractor
           
    create_topic_model(model='tf')
    
    
def visualize_topic_model():
    
    nmf_model, processed_data, feature_extractor = nmf_topic_model()
    pyLDAvis.enable_notebook()
    panel = pyLDAvis.sklearn.prepare(nmf_model, processed_data, feature_extractor, mds='tsne')
    panel
   
if __name__ == '__main__':    

    #sklearn_topic_model()
    #nmf_topic_model()
    gensim_topic_model()
    
