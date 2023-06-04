from keras.models import load_model
from util import *

maquina = "Linux" #remoto 
# maquina = "Windows" #local Albert y Miguel
#maquina = "MAC"

### DESCOMENTAR TU USUARIO EN LOCAL ###
#usuario = "34606"
usuario = "apuma"
#usuario = "carlosletaalfonso"

if maquina == "Linux":
    filename="/home/alumne/xnap-project-ed_group_03/output/char2encoding.pkl" #### remoto
elif maquina == "Windows":
    filename = os.path.join(r'C:\Users', usuario, r'github-classroom\DCC-UAB\xnap-project-ed_group_03\output\char2encoding.pkl') #### local 
else:
    filename = "/Users/carlosletaalfonso/github-classroom/DCC-UAB/xnap-project-ed_group_03/output/char2encoding.pkl" #### local leta

sentence="Hello"
#saveChar2encoding("char2encoding.pkl",input_token_index,16,71,reverse_target_char_index,num_decoder_tokens,target_token_index)
input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index= getChar2encoding(filename)
encoder_input_data=encodingSentenceToPredict(sentence,input_token_index,16,91) #(sentence,input_token_index,16,71)
encoder_model= load_model('encoder_modelPredTranslation.h5')
decoder_model= load_model('decoder_modelPredTranslation.h5')

input_seq = encoder_input_data

decoded_sentence=decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index)
print('-')
print('Input sentence:', sentence)
print('Decoded sentence:', decoded_sentence)
