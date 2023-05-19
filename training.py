from util import *

#maquina = "Linux" #remoto 
maquina = "Windows" #local Albert y Miguel
#maquina = "MAC"

### DESCOMENTAR TU USUARIO EN LOCAL ###
#usuario = "34606"
usuario = "apuma"
#usuario = "carlosletaalfonso"


#load the data and format  them for being processed
encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length=prepareData(data_path)

# we build the model
model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense=modelTranslation(num_encoder_tokens,num_decoder_tokens)

# we train it
trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data)

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
