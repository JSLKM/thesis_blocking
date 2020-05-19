import time

def _sentence_embedding(table, key_values):
    if key_values['embedding_type'] == 'doc2vec':
        from .doc2vec import tuple_doc2vec_embedding
        return tuple_doc2vec_embedding(table, key_values['attributes_list'])

    elif key_values['embedding_type'] == 'word2vec':
        from .word2vec import tuple_word2vec_embedding
        return tuple_word2vec_embedding(table, key_values['attributes_list'])

    elif key_values['embedding_type'] == 'inferSent':
        from .inferSent import tuple_inferSent_embedding
        return tuple_inferSent_embedding(
            table,
            model_type=key_values['model_type'],
            char_level=key_values['char_level'],
            model_version=key_values['model_version'],
            rnn_dim=key_values['rnn_dim'],
            verbose=key_values['verbose'],
            attributes_list=key_values['attributes_list'])

    elif key_values['embedding_type'] == 'glove':
        from .glove import tuple_glove_embedding
        return tuple_glove_embedding(table, key_values['attributes_list'])
    
    elif key_values['embedding_type'] == 'fastText':
        from .fastText import tuple_fastText_embedding
        return tuple_fastText_embedding(table, key_values['attributes_list'])
    
    elif key_values['embedding_type'] == 'wiki2vec':
        from .wiki2vec import tuple_wiki2vec_embedding
        return tuple_wiki2vec_embedding(table, key_values['attributes_list'])

def sentence_embedding(table, key_values):
    start_time = time.time()
    embeddings = _sentence_embedding(table, key_values)

    if key_values['verbose'] > 0:
        print("Embedding time is: {0}".format(time.time() - start_time))
    
    return embeddings