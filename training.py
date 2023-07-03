from util import *
from keras.utils import Sequence
import tensorflow as tf
import time
from tensorflow.keras.callbacks import EarlyStopping
import random

# Crear EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5) #que pare si la val_loss no cambia en 5 épocas

start_time = time.time()

def data_generator_aleatoritzant(data_path, batch_size):
    start_index = 0
    while True:
        # Load a batch of data
        encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length=prepareData(data_path, start_index=start_index, batch_size=batch_size)
        
        # Aleatoritzamos el orden de los datos
        data_size = len(encoder_input_data)
        order = np.random.permutation(data_size)
        encoder_input_data = encoder_input_data[order]
        decoder_input_data = decoder_input_data[order]
        decoder_target_data = decoder_target_data[order]

        # Yield the batch of data
        yield [encoder_input_data, decoder_input_data], decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length

        # Update the start index for the next batch
        start_index += batch_size


def data_generator_basic(data_path, batch_size):
    start_index = 0
    while True:
        # Load a batch of data
        encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length=prepareData(data_path, start_index=start_index, batch_size=batch_size)
        # Yield the batch of data
        yield [encoder_input_data, decoder_input_data], decoder_target_data,input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length
        
        # Update the start index for the next batch
        start_index += batch_size


maquina = "Linux" #remoto 
# maquina = "Windows" #local Albert y Miguel
# maquina = "MAC"

### DESCOMENTAR TU USUARIO EN LOCAL ###
#usuario = "34606"
#usuario = "apuma"
usuario = "carlosletaalfonso"

start_index = 0
batch_size = 30000
epochs = 10
steps = 4

encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length=prepareData(data_path)

# construïm el model amb totes les dades
model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense=modelTranslation(num_encoder_tokens,num_decoder_tokens)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=["accuracy"])

# DATA LOADER
generator = data_generator_aleatoritzant(data_path, batch_size)

for epoch in range(9999): #epochs
    for step in range(steps): #steps
        # Load the next batch of data from the generator
        data_batch = next(generator)
        encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length = data_batch[0][0], data_batch[0][1], data_batch[1], data_batch[2], data_batch[3], data_batch[4], data_batch[5], data_batch[6], data_batch[7], data_batch[8]
        # Train the model with the batch of data
        print("ÈPOCA:", epoch)
        trainSeq2Seq(model, encoder_input_data, decoder_input_data, decoder_target_data)

    # Verificar el tiempo transcurrido
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1200: #media hora,, 3600 <- una hora:  # Detener después de 1 hora
        break


    # # Realizar verificación temprana
    # if early_stopping.should_stop():
    #     break

# we build the final model for the inference (slightly different) and we save it
encoder_model,decoder_model,reverse_target_char_index=generateInferenceModel(encoder_inputs, encoder_states,input_token_index,target_token_index,decoder_lstm,decoder_inputs,decoder_dense)

# we save the object to convert the sequence to encoding  and encoding to sequence
# our model is made for being used with different langages that do not have the same number of letters and the same alphabet

if maquina == "Linux":
    filename="/home/alumne/xnap-project-ed_group_03/output/char2encoding.pkl" #### remoto
elif maquina == "Windows":
    filename = os.path.join(r'C:\Users', usuario, r'github-classroom\DCC-UAB\xnap-project-ed_group_03\output\char2encoding.pkl') #### local 
else:
    filename = "/Users/carlosletaalfonso/github-classroom/DCC-UAB/xnap-project-ed_group_03/output/char2encoding.pkl" #### local leta

saveChar2encoding(filename,input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index)
