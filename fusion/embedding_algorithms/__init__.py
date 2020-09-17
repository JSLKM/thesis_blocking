import time

def _sentence_embedding(table, key_values):
    if key_values['embedding_type'] == 'inferSent':
        from .inferSent import tuple_inferSent_embedding
        return tuple_inferSent_embedding(
            table,
            key_values)

def sentence_embedding(table, key_values):
    start_time = time.time()
    embeddings = _sentence_embedding(table, key_values)

    if key_values['verbose'] > 0:
        print('embedding_type: {0}'.format(key_values['embedding_type']))
        print('attributes_list: {0}'.format(key_values['attributes_list']))
        print('model_type: {0}'.format(key_values['model_type']))
        print('char_level: {0}'.format(key_values['char_level']))
        print("Embedding time is: {0}".format(time.time() - start_time))
    
    return embeddings

def set_embedding_model(key_values):
    start_time = time.time()
    from .inferSent import set_RNN_embedding
    set_RNN_embedding(key_values)
    if key_values['verbose'] > 0:
        print('model_version: {0}'.format(key_values['model_version']))
        print('rnn_dim: {0}'.format(key_values['rnn_dim']))
        print('model_type: {0}'.format(key_values['model_type']))
        print('char_level: {0}'.format(key_values['char_level']))
        print("Setup time is: {0}".format(time.time() - start_time))