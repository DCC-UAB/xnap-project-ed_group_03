from util_copy import *
import gc
from keras.utils import Sequence
import tensorflow as tf

def data_generator(data_path, batch_size):
    start_index = 0
    while True:
        # Load a batch of data
        encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,_,_,max_encoder_seq_length=prepareData(data_path, start_index=start_index, batch_size=batch_size)
        # Yield the batch of data
        yield [encoder_input_data, decoder_input_data], decoder_target_data
        
        # Update the start index for the next batch
        start_index += batch_size


maquina = "Linux" #remoto 
# maquina = "Windows" #local Albert y Miguel
#maquina = "MAC"

### DESCOMENTAR TU USUARIO EN LOCAL ###
#usuario = "34606"
usuario = "apuma"
#usuario = "carlosletaalfonso"

start_index = 0
batch_size = 20000

encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length=prepareData(data_path)

# construïm el model amb totes les dades
model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense=modelTranslation(num_encoder_tokens,num_decoder_tokens)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# DATA LOADER
generator = data_generator(data_path, batch_size)


for epoch in range(5):
    # del encoder_input_data, decoder_input_data, decoder_target_data
    # gc.collect()
    for step in range(5):        
        # Load the next batch of data from the generator
        data_batch = next(generator)
        encoder_input_data, decoder_input_data, decoder_target_data = data_batch[0][0], data_batch[0][1], data_batch[1]        
        # Train the model with the batch of data
        print("ÈPOCA:", epoch)
        trainSeq2Seq(model, encoder_input_data, decoder_input_data, decoder_target_data, epoch) # li passem l'epoch per a calcular el lr


# Determina el número total de lotes y el tamaño del lote
# num_batches = 12

# for epoch in range(10):
#     start_index = 0    
#     for step in range(num_batches):
#         print("ÉPOCA:", epoch)
#         # Carga el lote actual en la memoria principal (CPU)
#         batch_encoder_input_data = encoder_input_data[start_index: start_index + batch_size]
#         batch_decoder_input_data = decoder_input_data[start_index: start_index + batch_size]
#         batch_decoder_target_data = decoder_target_data[start_index: start_index + batch_size]
        
#         # Transfiere los lotes a la GPU
#         # batch_encoder_input_data_gpu = tf.convert_to_tensor(batch_encoder_input_data)
#         # batch_decoder_input_data_gpu = tf.convert_to_tensor(batch_decoder_input_data)
#         # batch_decoder_target_data_gpu = tf.convert_to_tensor(batch_decoder_target_data)
        
#         # Entrena el modelo con el lote actual en la GPU
#         trainSeq2Seq(model, batch_encoder_input_data, batch_decoder_input_data, batch_decoder_target_data, epoch)
        
#         # Libera la memoria de los lotes en la GPU
#         # del batch_encoder_input_data_gpu
#         # del batch_decoder_input_data_gpu
#         # del batch_decoder_target_data_gpu
        
#         # Incrementa el índice de inicio para el siguiente lote
#         start_index += batch_size


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
