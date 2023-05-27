from util_copy import *

class DataLoader:
    def __init__(self, encoder_input_data, decoder_input_data, decoder_target_data, batch_size):
        self.encoder_input_data = encoder_input_data
        self.decoder_input_data = decoder_input_data
        self.decoder_target_data = decoder_target_data
        self.batch_size = batch_size

    def generate_batches(self):
        num_batches = len(self.encoder_input_data) // self.batch_size
        while True:
            for i in range(num_batches):
                start_index = i * self.batch_size
                end_index = start_index + self.batch_size
                yield ([self.encoder_input_data[start_index: end_index],
                        self.decoder_input_data[start_index: end_index]],
                        self.decoder_target_data[start_index: end_index])


maquina = "Linux" #remoto 
# maquina = "Windows" #local Albert y Miguel
#maquina = "MAC"

### DESCOMENTAR TU USUARIO EN LOCAL ###
#usuario = "34606"
#usuario = "apuma"
#usuario = "carlosletaalfonso"

start_index = 0
batch_size = 30000

encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length=prepareData(data_path)

# construïm el model amb totes les dades
model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense=modelTranslation(num_encoder_tokens,num_decoder_tokens)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# DATA LOADER
data_loader = DataLoader(encoder_input_data, decoder_input_data, decoder_target_data, batch_size)

# start_index += batch_size

'''
for i in range(40): # Fem 10 epochs, en cada epoch es llegeixen els 4 blocs
    #load the data and format  them for being processed
    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,_,_,max_encoder_seq_length=prepareData(data_path, start_index=start_index, batch_size=batch_size)
    # we train it
    trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data, i) # li passem l'epoch per a calcular el lr
    if start_index == 120000:
        start_index = 0
    else:
        start_index += batch_size
'''

for i in range(40):
    for batch in data_loader.generate_batches():
        # Assuming that trainSeq2Seq can take a full batch as argument
        encoder_input, decoder_input = batch[0]
        decoder_target = batch[1]
        trainSeq2Seq(model, encoder_input, decoder_input, decoder_target, i)

    if start_index == 120000:
        start_index = 0
    else:
        start_index += batch_size

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
