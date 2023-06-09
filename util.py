from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU, Dropout, BatchNormalization
from keras.models import load_model
from keras import regularizers
import numpy as np
import pickle
import os
import wandb
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras.initializers import glorot_normal
from keras.regularizers import L1L2


batch = 256  # Batch size for training.
epochs_batch = 1  # Number of epochs to train each batch for.
latent_dim = 512  # Latent dimensionality of the encoding space. 1024
num_samples = 130000 # Number of samples to train on. 139705
num_hidden_units = 110

decay_rate = 0.5 # reduction factor of learning rate
decay_step =  25 # number of epochs for learning rate decay


#maquina = "Linux" #remoto 
#maquina = "Windows" #local Albert y Miguel
maquina = "MAC"

# Path to the data txt file on disk.
#data_path = '/home/alumne/xnap-project-ed_group_03/spa.txt' # to replace by the actual dataset name
data_path = 'spa-eng/spa.txt'
encoder_path='encoder_modelPredTranslation.h5'
decoder_path='decoder_modelPredTranslation.h5'

### Sustituir por su usuario ###
#usuario = "34606"
#usuario = "apuma"
usuario = "carlosletaalfonso"

if maquina == "Linux":
    LOG_PATH="/home/alumne/xnap-project-ed_group_03/log" #### remoto
elif maquina == "Windows":
    LOG_PATH = os.path.join(r'C:\Users', usuario, r'github-classroom\DCC-UAB\xnap-project-ed_group_03\log') #### local 
else:
    LOG_PATH = "/Users/carlosletaalfonso/github-classroom/DCC-UAB/xnap-project-ed_group_03/log" #### local leta



# IF USING CALLBACKS, USE THIS FUNCTIONS
# def schedule_learning_rate(epoch): #exponencial
#     lr = 0.01 * 0.001 ** epoch
#     return lr

# def scheduler_decay(epoch, lr): # decay
#     decay_rate = 0.1 
#     decay_step =  8
#     print("LEARNING RATE: ", lr)
#     if epoch % decay_step == 0 and epoch:
#         return lr * decay_rate
#     return max(lr, 0.001)  # Aquí es el learning rate mínimo 
    



def prepareData(data_path, start_index=None, batch_size=None):
    if batch_size:
        input_characters,target_characters,input_texts,target_texts=extractChar_batch(data_path, start_index, batch_size)
    else:
        input_characters,target_characters,input_texts,target_texts=extractChar(data_path)
    
    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length = encodingChar(input_characters,target_characters,input_texts,target_texts)
    
    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length

def extractChar(data_path,exchangeLanguage=False):
    # We extract the data (Sentence1 \t Sentence 2) from the anki text file
    input_texts = [] # seqüencies (frases) per traduir
    target_texts = [] # seqüencies traduides
    input_characters = set() # caracters o simbols únics de les seqüencies per traduir
    target_characters = set() # caracters o simbols únics de les seqüencies traduides
    lines = open(data_path, encoding='utf-8').read().split('\n')
    print(str(len(lines) - 1))
    if (exchangeLanguage==False):
        for line in lines[: min(num_samples, len(lines) - 1)]:
            input_text, target_text = line.split('\t')[0], line.split('\t')[1]
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))

    else:
        for line in lines[: min(num_samples, len(lines) - 1)]:
            target_text , input_text = line.split('\t')
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))

    return input_characters,target_characters,input_texts,target_texts

def extractChar_batch(data_path, input_characters, target_characters,exchangeLanguage=False, start_index=0, batch_size=20000):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    lines = open(data_path, encoding='utf-8').read().split('\n')
    print(str(len(lines) - 1))

    num_samples = min(start_index + batch_size, len(lines) - 1) 

    if (exchangeLanguage == False):
        for line in lines[start_index:num_samples]:
            input_text, target_text = line.split('\t')[0], line.split('\t')[1]
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

    else:
        for line in lines[start_index:num_samples]:
            target_text, input_text, _ = line.split('\t')
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))

    return input_characters, target_characters, input_texts, target_texts

    
def encodingChar(input_characters,target_characters,input_texts,target_texts):
# We encode the dataset in a format that can be used by our Seq2Seq model (hot encoding).
# Important: this project can be used for different language that do not have the same number of letter in their alphabet.
# Important2: the decoder_target_data is ahead of decoder_input_data by one timestep (decoder = LSTM cell).
# 1. We get the number of letter in language 1 and 2 (num_encoder_tokens/num_decoder_tokens)
# 2. We create a dictonary for both language
# 3. We store their encoding and return them and their respective dictonary
    
    num_encoder_tokens = 91 #len(input_characters) # numero total de caracters unics en les seqüencies de entrada (util per dimensio vocetor one hot, per representar cada caracter en seqüencies de entrada)
    num_decoder_tokens = 110 #len(target_characters) # numero total de caracters unics en les seqüencies de sortida
    max_encoder_seq_length = max([len(txt) for txt in input_texts]) # longitud de la seqüencia de entrada més llarga (pasos de temps tindra la xarxa)
    max_decoder_seq_length = max([len(txt) for txt in target_texts]) # longitud de la seqüencia de sortida més llarga
    print('Number of num_encoder_tokens:', num_encoder_tokens)
    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)]) # mapea cada caracter únic en la seqüencia d'entrada a un índex numeric únic
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)]) # mapea cada caracter únic en la seqüencia de sortida a un índex numeric únic

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens),dtype='float32') # matriu seqüencies entrada codificades (one hot). La matriu és: cada bloc una seqüencia, tantes files com seqüencia mes llarga, tantes columnes com total de caracters 
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32') # matriu seqüencies sortida codificades (one hot)
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32') # mateix que decoder_input_data pero desplaçada un cap cap endavant

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1. # assigna un 1 a les posiciones dels caracters, la resta de la fila tot 0
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.


    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length
	
def modelTranslation2(num_encoder_tokens,num_decoder_tokens):
# We crete the model 1 encoder(gru) + 1 decode (gru) + 1 Dense layer + softmax

    encoder_inputs = Input(shape=(None, num_encoder_tokens)) # entrada codificador
    encoder = GRU(latent_dim, return_state=True) # codificador com a capa GRU
    encoder_outputs, state_h = encoder(encoder_inputs) # obtenim sortida codificador i ultim estat intern
    encoder_states = state_h # guardem estat intern per després usar en deco

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_gru = GRU(latent_dim, return_sequences=True)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h) # estat inicial deco = ultim estat codificador
    decoder_dense = Dense(num_decoder_tokens, activation='softmax') # definim capa densa (totalment conectada) i func. act. softmax
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_gru,decoder_dense
	
def modelTranslation(num_encoder_tokens, num_decoder_tokens):
    # codificador
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_sequences=True, return_state=True, kernel_initializer=glorot_normal(seed=None), kernel_regularizer=regularizers.l2(0.001))
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    encoder_outputs = BatchNormalization()(encoder_outputs)  # Añadir Batch Normalization

    # capa lstm extra codificador 2
    encoder_outputs, state_h2, state_c2 = LSTM(latent_dim, return_sequences=True, return_state=True)(encoder_outputs)
    encoder_states2 = [state_h2, state_c2]
    encoder_outputs = BatchNormalization()(encoder_outputs)  # Añadir Batch Normalization

    # capa lstm extra codificador 3
    encoder_outputs, state_h3, state_c3 = LSTM(latent_dim, return_sequences=True, return_state=True)(encoder_outputs)
    encoder_states3 = [state_h3, state_c3]
    encoder_outputs = BatchNormalization()(encoder_outputs)  # Añadir Batch Normalization

    # decodificador
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, kernel_initializer=glorot_normal(seed=None), kernel_regularizer=regularizers.l2(0.001))
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    #decoder_outputs = BatchNormalization()(decoder_outputs)  # Añadir Batch Normalization

    # capa lstm extra decodificador 2
    decoder_outputs, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(decoder_outputs, initial_state=encoder_states2)
    #decoder_outputs = BatchNormalization()(decoder_outputs)  # Añadir Batch Normalization

    # capa lstm extra decodificador 3
    decoder_outputs, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(decoder_outputs, initial_state=encoder_states3)
    #decoder_outputs = BatchNormalization()(decoder_outputs)  # Añadir Batch Normalization

    # capa densa extra
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)


    # decoder_outputs = Dropout(0.2)(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model, decoder_outputs, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense




def trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data, epoch, lr):
# We load tensorboad
# We train the model
    if maquina == "Linux":
        LOG_PATH="/home/alumne/xnap-project-ed_group_03/output/log" #### remoto
    elif maquina == "Windows":
        LOG_PATH = os.path.join(r'C:\Users', usuario, r'github-classroom\DCC-UAB\xnap-project-ed_group_03\output\log') #### local 
    else:
        LOG_PATH = "/Users/carlosletaalfonso/github-classroom/DCC-UAB/xnap-project-ed_group_03/output/log" #### local leta
        
    # LOG IN TO WANDB WITH YOUR API KEY
    #api_key = "XXXXXXXXXXXXXXXXXX" # your api key
    #wandb.login(key = api_key)
    wandb.init(project="XNAP-PROJECT-ED_GROUP_03")
    wandb_callback = wandb.keras.WandbCallback()
    
    wandb.config.batch_size = batch
    wandb.config.epochs = epochs_batch
    wandb.config.validation_split = 0.05

    # IF USING CALLBAKCS, USE THIS CODE
    #lr_scheduler = LearningRateScheduler(schedule_learning_rate) #exponencial
    #lr_scheduler = LearningRateScheduler(scheduler_decay) #decay
    #Plateau lr
    #lr_scheduler = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.1, patience = 5, verbose = 1, min_lr = 0.00001) #0.5, 10, .0001


    # IN CODE IMPLEMENTED LEARNING RATE SCHEDULER (WITH FACTOR DECAY)
    print("LEARNING RATE: ", lr)
    if epoch % decay_step == 0 and epoch:
        lr *= decay_rate
    lr = max(lr, 0.00001) #/ Aquí es el learning rate mínimo 

    model.optimizer.learning_rate =  lr

    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch,
              epochs=epochs_batch,
              validation_split=0.05,
              callbacks = [wandb_callback]) #,lr_scheduler]) # IF USING CALLBACKS, UNCOMMENT THIS
    
    return lr
    
def generateInferenceModel(encoder_inputs, encoder_states,input_token_index,target_token_index,decoder_lstm,decoder_inputs,decoder_dense):
# Once the model is trained, we connect the encoder/decoder and we create a new model
# Finally we save everything
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())
    encoder_model.save(encoder_path)
    decoder_model.save(decoder_path)
    return encoder_model,decoder_model,reverse_target_char_index

def loadEncoderDecoderModel():
# We load the encoder model and the decoder model and their respective weights
    encoder_model= load_model(encoder_path)
    decoder_model= load_model(decoder_path)
    return encoder_model,decoder_model

def decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index):
# We run the model and predict the translated sentence
    # We encode the input
    states_value = encoder_model.predict(input_seq)
    
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    # We predict the output letter by letter 
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # We translate the token in hamain language
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # We check if it is the end of the string
        if (sampled_char == '\n' or
           len(decoded_sentence) > 500):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence

def encodingSentenceToPredict(sentence,input_token_index,max_encoder_seq_length,num_encoder_tokens):
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    for t, char in enumerate(sentence):
        encoder_input_data[0, t, input_token_index[char]] = 1.
    return encoder_input_data

def saveChar2encoding(filename,input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index):
    f = open(filename, "wb")
    pickle.dump(input_token_index, f)
    pickle.dump(max_encoder_seq_length, f)
    pickle.dump(num_encoder_tokens, f)
    pickle.dump(reverse_target_char_index, f)
    
    pickle.dump(num_decoder_tokens, f)
    
    pickle.dump(target_token_index, f)
    f.close()
    

def getChar2encoding(filename):
    f = open(filename, "rb")
    input_token_index = pickle.load(f)
    max_encoder_seq_length = pickle.load(f)
    num_encoder_tokens = pickle.load(f)
    reverse_target_char_index = pickle.load(f)
    num_decoder_tokens = pickle.load(f)
    target_token_index = pickle.load(f)
    f.close()
    return input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index
