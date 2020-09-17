import numpy as np
import nltk
import os
import copy
import torch

# the pretrained model we are using
lstm_embedder_model = None
bi_lstm_embedder_model = None

def set_lstm_embedder_model(key_values):

    char_level_embedding = key_values['char_level']
    model_version = key_values['model_version']
    rnn_dim = key_values['rnn_dim']

    from .InferSent.models import LSTMEncoder

    global lstm_embedder_model
    dirname = os.path.dirname(__file__)
    model_version = model_version
    model_name = 'char_level/cc/LSTM' + str(rnn_dim) + '/LSTM_' + str(rnn_dim) + '_char_cc.pickle.encoder.pkl' if char_level_embedding else 'word_level/glove/LSTM' + str(rnn_dim) + '/LSTM_' + str(rnn_dim) + '_word_glove.pickle.encoder.pkl'
    MODEL_PATH = os.path.join(dirname, './InferSent/trained_models/SNLI_corpus/'+model_name)

    params_model = {
        'bsize': 64, 
        'word_emb_dim': 300, 
        'enc_lstm_dim': rnn_dim,
        'pool_type': 'max', 
        'dpout_model': 0.0, 
        'version': model_version
    }

    lstm_embedder_model = LSTMEncoder(params_model)
    lstm_embedder_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    EMBED_PATH = os.path.join(dirname, '../../embedding_pretrained_models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin' if char_level_embedding else '../../embedding_pretrained_models/glove.840B.300d/glove.840B.300d.txt')

    if char_level_embedding:
        lstm_embedder_model.set_char_level_embedding_model()
    else:
        lstm_embedder_model.set_w2v_path(EMBED_PATH)
        lstm_embedder_model.build_vocab_k_words(K=2196017) # Glove Size

def set_bi_lstm_embedder_model(key_values):

    char_level_embedding = key_values['char_level']
    model_version = key_values['model_version']
    rnn_dim = key_values['rnn_dim']

    from .InferSent.models import InferSent

    global bi_lstm_embedder_model
    dirname = os.path.dirname(__file__)
    model_version = model_version
    model_name = 'char_level/cc/INFERSENT'+str(rnn_dim)+'/INFERSENT_'+str(rnn_dim)+'_char_cc.pickle.encoder.pkl' if char_level_embedding else 'word_level/glove/INFERSENT'+str(rnn_dim)+'/INFERSENT_'+str(rnn_dim)+'_word_glove.pickle.encoder.pkl'
    MODEL_PATH = os.path.join(dirname, './InferSent/trained_models/SNLI_corpus/'+model_name)
    params_model = {
        'bsize': 64, 
        'word_emb_dim': 300, 
        'enc_lstm_dim': rnn_dim,
        'pool_type': 'max', 
        'dpout_model': 0.0, 
        'version': model_version
    }

    bi_lstm_embedder_model = InferSent(params_model)
    bi_lstm_embedder_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    EMBED_PATH = os.path.join(dirname, '../../embedding_pretrained_models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin' if char_level_embedding else '../../embedding_pretrained_models/glove.840B.300d/glove.840B.300d.txt')
    if char_level_embedding:
        bi_lstm_embedder_model.set_char_level_embedding_model()
    else:
        bi_lstm_embedder_model.set_w2v_path(EMBED_PATH)
        bi_lstm_embedder_model.build_vocab_k_words(K=2196017)

def set_RNN_embedding(key_values):
    if key_values['model_type'] == 'lstm':
        set_lstm_embedder_model(key_values)
    elif key_values['model_type'] == 'bilstm':
        set_bi_lstm_embedder_model(key_values)

def RNN_embedding(table, attributes_list, model_type, char_level):

    if attributes_list == []:
        print('all attributes used')
        row = table.loc[0]
        attributes_list = row.index

    row_string = []
    sentences = []
    for index in table.index:  # Each Row
        row = table.loc[index]
        for attribute in attributes_list:  # Attributes of each Row
            attribute_value = row.loc[attribute]
            row_string.append(str(attribute_value))
        sentences.append(' '.join(row_string))
        row_string.clear()

    # Send directly all the embedding
    if model_type == 'lstm':
        if char_level:
            return lstm_embedder_model.encode_char_level(sentences, bsize=64, tokenize=False, verbose=False)
        else:
            return lstm_embedder_model.encode_word_level(sentences, bsize=64, tokenize=False, verbose=False)
    elif model_type == 'bilstm':
        if char_level:
            return bi_lstm_embedder_model.encode_char_level(sentences, bsize=64, tokenize=False, verbose=False)
        else:
            return bi_lstm_embedder_model.encode_word_level(sentences, bsize=64, tokenize=False, verbose=False)

def tuple_inferSent_embedding(table, key_values):
    embeddings = [] 
    
    embeddings = RNN_embedding(table, key_values['attributes_list'], key_values['model_type'], key_values['char_level'])
    embeddings = np.array(embeddings)
    return embeddings
