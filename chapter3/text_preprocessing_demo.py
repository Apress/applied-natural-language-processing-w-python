#Chapter 3 text preprocessing examples 
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np
from textblob import TextBlob as tb


sample_text = '''I am a student from the University of Alabama. I 
was born in Ontario, Canada and I am a huge fan of the United States. 
I am going to get a degree in Philosophy to improve my chances of 
becoming a Philosophy professor. I have been working towards this goal
for 4 years. I am currently enrolled in a PhD program. It is very difficult, 
but I am confident that it will be a good decision'''

print(sample_text)

from nltk.tokenize import word_tokenize, sent_tokenize

sample_word_tokens = word_tokenize(sample_text)
print(sample_word_tokens)
sample_sent_tokens = sent_tokenize(sample_text)
print(sample_sent_tokens)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)

def mistake():
    stop_words = stopwords.words('english')
    word_tokens = [word for word in sample_word_tokens if word not in stop_words]
    print(word_tokens)
    return word_tokens
    
mistake = mistake()
len(mistake)

def advised_preprocessing(sample_word_tokens=sample_word_tokens):
    stop_words = [word.upper() for word in stopwords.words('english')]
    word_tokens = [word for word in sample_word_tokens if word.upper() not in stop_words]
    print(word_tokens)
    return word_tokens

sample_word_tokens = advised_preprocessing()

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
sample_word_tokens = tokenizer.tokenize(str(sample_word_tokens))
sample_word_tokens = [word.lower() for word in sample_word_tokens]
print(sample_word_tokens)

import collections, re

def bag_of_words(text):
    _bag_of_words = [collections.Counter(re.findall(r'\w+', word)) for word in text]
    bag_of_words = sum(_bag_of_words, collections.Counter())
    return bag_of_words
    
sample_word_tokens_bow = bag_of_words(text=sample_word_tokens)
print(sample_word_tokens_bow)

sample_text_bow = bag_of_words(text=word_tokenize(sample_text))

from sklearn.feature_extraction.text import CountVectorizer

def bow_sklearn(text=sample_sent_tokens):
    c = CountVectorizer(stop_words='english', token_pattern=r'\w+')
    converted_data = c.fit_transform(text).todense()
    print(converted_data.shape)
    return converted_data, c.get_feature_names()

bow_data, feature_names = bow_sklearn()
print(bow_data); print(feature_names)

'''
TF-IDF EXAMPLES
'''

text = tb('''I was a student at the University of Pennsylvania, but now work on 
Wall Street as a Lawyer. I have been living in New York for roughly five years
now, however I am looking forward to eventually retiring to Texas once I have 
saved up enough money to do so.''')


text2= tb('''I am a doctor who is considering retirement in the next couple of years. 
I went to the Yale University, however that was quite a long time ago. I have two children,
who both have three children each, making me a grandfather. I look forward to retiring 
and spending more time with them''')

def tf_idf_example(textblobs=[text, text2]):
    
    def term_frequency(word, textblob):
        return textblob.words.count(word)/float(len(textblob.words))
    
    def document_counter(word, text):
        return sum(1 for blob in text if word in blob)
    
    def idf(word, text):
        return np.log(len(text) /1 + float(document_counter(word, text)))
    
    def tf_idf(word, blob, text):
        return term_frequency(word, blob) * idf(word, text)

    output = list()
    for i, blob in enumerate(textblobs):
        output.append({word: tf_idf(word, blob, textblobs) for word in blob.words})
        
    print(output)

tf_idf_example()


from sklearn.feature_extraction.text import TfidfVectorizer

text = '''I was a student at the University of Pennsylvania, but now work on 
Wall Street as a Lawyer. I have been living in New York for roughly five years
now, however I am looking forward to eventually retiring to Texas once I have 
saved up enough money to do so.'''

text2= '''I am a doctor who is considering retirement in the next couple of years. 
I went to the Yale University, however that was quite a long time ago. I have two children,
who both have three children each, making me a grandfather. I look forward to retiring 
and spending more time with them'''

def tf_idf_sklearn(document_list=list([text, text2])):
    t = TfidfVectorizer(stop_words='english', token_pattern=r'\w+')
    x = t.fit_transform(document_list).todense()
    print(x)
    print(x.shape)
    
tf_idf_sklearn()
