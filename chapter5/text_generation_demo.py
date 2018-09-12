#Text Generation Demo
#Taweh Beysolow II 

#Import the necessary modules
import numpy as np
from chapter4.word_embeddings import load_data
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from keras.utils import np_utils

max_pages = 30
pdf_file = 'harry_potter.pdf'
misc = '''... '' -- '''.split()
sequence_length = 100
window_size = 2

def preprocess_data(sequence_length=sequence_length, max_pages=max_pages, pdf_file=pdf_file):
    text_data = load_data(max_pages=max_pages, pdf_file=pdf_file)
    characters = list(set(text_data.lower()))
    character_dict = dict((character, i) for i, character in enumerate(characters))
    int_dictionary = dict((i, character) for i, character in enumerate(characters))
    num_chars, vocab_size = len(text_data), len(characters)
    x, y = [], []

    for i in range(0, num_chars - sequence_length, 1):
        input_sequence = text_data[i: i+sequence_length]
        output_sequence = text_data[i+sequence_length]
        x.append([character_dict[character.lower()] for character in input_sequence])
        y.append(character_dict[output_sequence.lower()])
    
    for k in range(0, len(x)): x[i] = [_x for _x in x[i]]    
    x = np.reshape(x, (len(x), sequence_length, 1))
    x, y = x/float(vocab_size), np_utils.to_categorical(y)
    return x, y, num_chars, vocab_size, int_dictionary
    
def train_rnn_keras(epochs, activation, num_units): 
    
    x, y, num_chars, vocab_size, int_dictionary = preprocess_data()
    
    def create_rnn(num_units=num_units, activation=activation):
        model = Sequential()
        model.add(LSTM(num_units, activation=activation, input_shape=(None, x.shape[1])))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')            
        model.summary()
        return model
            
    rnn_model = create_rnn()
    _x = x.reshape(x.shape[0], 1, x.shape[1])
    rnn_model.fit(_x, y, epochs=epochs, shuffle=True)
        
    #Generating text from neural network
    predictions = rnn_model.predict(_x[1:])
    predictions = [np.argmax(prediction) for prediction in predictions]
    text = [int_dictionary[index] for index in predictions]
    print(''.join([word for word in text])) 
    

def train_brnn_keras(epochs, activation, num_units):
        
    x, y, num_chars, vocab_size, int_dictionary = preprocess_data()
    
    def create_rnn(num_units=num_units, activation=activation):
        model = Sequential()
        
        model.add(Bidirectional(LSTM(num_units, activation=activation),
                                input_shape=(None, x.shape[1])))
        
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')            
        model.summary()
        return model
            
    rnn_model = create_rnn()
    _x = x.reshape(x.shape[0], 1, x.shape[1])
    rnn_model.fit(_x, y, epochs=epochs, shuffle=True)
        
    #Generating text from neural network
    predictions = rnn_model.predict(_x[1:])
    predictions = [np.argmax(prediction) for prediction in predictions]
    text = [int_dictionary[index] for index in predictions]
    print(''.join([word for word in text])) 

    
if __name__ == '__main__':
    
    train_rnn_keras(epochs=300, num_units=300, activation='selu')
    #train_brnn_keras(epochs=100, num_units=200, activation='relu')
