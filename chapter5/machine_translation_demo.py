#Machine Translation Demo 
# -*- coding: utf-8 -*-
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np
from keras.models import Model, Input
from keras.layers import LSTM, Dense 
 
#Parameters
n_units = 400; epochs = 1000; 
batch_size = 50; max_pairs = 10000

def remove_non_ascii(text):
    return ''.join([word for word in text if ord(word) < 128])

def load_data():
    input_characters, output_characters = set(), set()
    input, output = [], []
    sentence_pairs = open('/Users/tawehbeysolow/Downloads/deu-eng/deu.txt').read().split('\n')
    for line in sentence_pairs[: min(max_pairs, len(sentence_pairs))-1]:
        _input, _output = line.split('\t')
        output.append(_output)
        input.append(_input)
        for i in _input: 
            if i not in input_characters: input_characters.add(i.lower())
        for o in _output:
            if o not in output_characters: output_characters.add(o.lower())
            
    input_characters = sorted(list(input_characters))
    output_characters = sorted(list(output_characters))
    n_encoder_tokens, n_decoder_tokens = len(input_characters), len(output_characters)
    max_encoder_len = max([len(text) for text in input])
    max_decoder_len = max([len(text) for text in output])    
    input_dictionary = {word: i for i, word in enumerate(input_characters)}
    output_dictionary = {word: i for i, word in enumerate(output_characters)}
    label_dictionary = {i: word for i, word in enumerate(output_characters)}
    x_encoder = np.zeros((len(input), max_encoder_len, n_encoder_tokens), dtype=float)
    x_decoder = np.zeros((len(input), max_decoder_len, n_decoder_tokens), dtype=float)
    y_decoder = np.zeros((len(input), max_decoder_len, n_decoder_tokens), dtype=float)
    
    for i, (_input, _output) in enumerate(zip(input, output)):
        for _character, character in enumerate(_input):
            x_encoder[i, _character, input_dictionary[character.lower()]] = 1
        for _character, character in enumerate(_output):
            x_decoder[i, _character, output_dictionary[character.lower()]] = 1
            if _character > 0: y_decoder[i, _character-1, output_dictionary[character.lower()]] = 1

    data = list([x_encoder, x_decoder, y_decoder])      
    variables = list([label_dictionary, n_decoder_tokens, n_encoder_tokens])                             
    return data, variables
    
def encoder_decoder(n_encoder_tokens, n_decoder_tokens):
    
    encoder_input = Input(shape=(None, n_encoder_tokens))    
    encoder = LSTM(n_units, return_state=True)
    encoder_output, hidden_state, cell_state = encoder(encoder_input)
    encoder_states = [hidden_state, cell_state]
    
    decoder_input = Input(shape=(None, n_decoder_tokens))
    decoder = LSTM(n_units, return_state=True, return_sequences=True)
    decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_states)
    
    decoder = Dense(n_decoder_tokens, activation='softmax')(decoder_output)
    model = Model([encoder_input, decoder_input], decoder)
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    model.summary()
    return model
    
def train_encoder_decoder():
    
    input_data_objects = load_data()
    x_encoder, x_decoder, y_decoder = input_data_objects[0][0], input_data_objects[0][1], input_data_objects[0][2]
    label_dictionary, n_decoder_tokens = input_data_objects[1][0], input_data_objects[1][1]
    n_encoder_tokens = input_data_objects[1][2]

    seq2seq_model = encoder_decoder(n_encoder_tokens, n_decoder_tokens)
    seq2seq_model.fit([x_encoder, x_decoder], y_decoder, epochs=epochs, batch_size=batch_size, shuffle=True)
    
    #Comparing model predictions and actual labels
    for start, end in zip(range(10, 20, 1), range(11, 21, 1)):
        y_predict = seq2seq_model.predict([x_encoder[start:end], x_decoder[start:end]])
        input_sequences, output_sequences = [], []
        for i in range(0, len(y_predict[0])): 
            output_sequences.append(np.argmax(y_predict[0][i]))
            input_sequences.append(np.argmax(x_decoder[start][i]))
        
        output_sequences = ''.join([label_dictionary[key] for key in output_sequences])
        input_sequences = ''.join([label_dictionary[key] for key in input_sequences])
        print('Model Prediction: ' + output_sequences); print('Actual Output: ' + input_sequences)

if __name__ == '__main__':
    
    train_encoder_decoder()
    
