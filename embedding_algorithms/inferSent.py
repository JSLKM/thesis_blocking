import numpy as np
import nltk
import os
import copy
import torch

# the pretrained model we are using
lstm_embedder_model = None
bi_lstm_embedder_model = None

def set_lstm_embedder_model(char_level_embedding, model_version, rnn_dim):

    from InferSent.models import LSTMEncoder

    global lstm_embedder_model
    dirname = os.path.dirname(__file__)
    model_version = model_version
    model_name = 'char_level/cc/LSTM' + str(rnn_dim) + '/LSTM_' + str(rnn_dim) + '_char_cc.pickle.encoder.pkl' if char_level_embedding else 'word_level/glove/LSTM' + str(rnn_dim) + '/LSTM_' + str(rnn_dim) + '_word_glove.pickle.encoder.pkl'
    MODEL_PATH = os.path.join(dirname, '../InferSent/trained_models/SNLI_corpus/'+model_name)

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

    EMBED_PATH = os.path.join(dirname, '../embedding_pretrained_models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin' if char_level_embedding else '../embedding_pretrained_models/glove.840B.300d/glove.840B.300d.txt')

    if char_level_embedding:
        lstm_embedder_model.set_char_level_embedding_model()
    else:
        lstm_embedder_model.set_w2v_path(EMBED_PATH)
        lstm_embedder_model.build_vocab_k_words(K=2196017) # Glove Size


def set_bi_lstm_embedder_model(char_level_embedding, model_version, rnn_dim):
    from InferSent.models import InferSent

    global bi_lstm_embedder_model
    dirname = os.path.dirname(__file__)
    model_version = model_version
    model_name = 'char_level/cc/INFERSENT'+str(rnn_dim)+'/INFERSENT_'+str(rnn_dim)+'_char_cc.pickle.encoder.pkl' if char_level_embedding else 'word_level/glove/INFERSENT'+str(rnn_dim)+'/INFERSENT_'+str(rnn_dim)+'_word_glove.pickle.encoder.pkl'
    MODEL_PATH = os.path.join(dirname, '../InferSent/trained_models/SNLI_corpus/'+model_name)
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
    EMBED_PATH = os.path.join(dirname, '../embedding_pretrained_models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin' if char_level_embedding else '../embedding_pretrained_models/glove.840B.300d/glove.840B.300d.txt')
    if char_level_embedding:
        bi_lstm_embedder_model.set_char_level_embedding_model()
    else:
        bi_lstm_embedder_model.set_w2v_path(EMBED_PATH)
        bi_lstm_embedder_model.build_vocab_k_words(K=2196017)


def set_RNN_embedding(model_type, char_level, model_version, rnn_dim, verbose):
    if verbose > 1:
        print("set_RNN_embedding...")
    if model_type == 'lstm':
        set_lstm_embedder_model(char_level, model_version, rnn_dim)
    elif model_type == 'bilstm':
        set_bi_lstm_embedder_model(char_level, model_version, rnn_dim)


def RNN_embedding(table, model_type, char_level):

    row_string = []
    sentences = []
    for index in table.index:  # Each Row
        row = table.loc[index]
        for attribute in row.index:  # Attributes of each Row
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

# if m is #attributes in row and d the embedding dimension, return a m*d matrix

def tuple_inferSent_embedding(table, **kwargs):
    embeddings = [] 

    set_RNN_embedding(
        kwargs['model_type'], 
        kwargs['char_level'],
        kwargs['model_version'], 
        kwargs['rnn_dim'],
        kwargs['verbose'],
    )
    print("model_type: {0}".format(kwargs['model_type']))
    print("char_level: {0}".format(kwargs['char_level']))
    print("model_version: {0}".format(kwargs['model_version']))
    print("rnn_dim: {0}".format(kwargs['rnn_dim']))
    
    embeddings = RNN_embedding(table, kwargs['model_type'], kwargs['char_level'])
    embeddings = np.array(embeddings)
    # print_embeddings_to_file(embeddings,**kwargs)
    return embeddings


# def print_embeddings_to_file(embeddings, **kwargs):
#     if kwargs['composition_method'] == 'mean':
#         end_path = '/mean_based/cc_based/embeddings.npy' if kwargs[
#             'char_level'] else '/mean_based/glove_based/embeddings.npy'
#         path = '/home/marco/PycharmProjects/learning_thesis/thesis/tuple_embeddings/' + \
#             kwargs['dataset']+end_path
#         np.save(path, embeddings)
#     else:
#         # composition method is rnn based
#         name_file = '/embeddings.npy'
#         path = '/home/marco/PycharmProjects/learning_thesis/thesis'+name_file
#         np.save(path, embeddings)