import numpy as np
import gensim.models as g
import nltk
import os
import time
from tqdm import tqdm


word2vec_model = None


def set_word2vec_model():
    start_time = time.time()
    global word2vec_model
    dirname = os.path.dirname(__file__)
    path = "../embedding_pretrained_models/enwiki_20180420_300d/enwiki_20180420_300d.pkl"
    model_path = os.path.join(dirname, path)
    word2vec_model = g.KeyedVectors.load_word2vec_format(
        model_path, binary=False, unicode_errors='ignore')
    print("Model Setup time is: {0}".format(time.time() - start_time))


def tuple_word2vec_embedding(table, attributes_list):
    # List of lists, each contained list is the embedding of the tuple
    embeddings = []
    set_word2vec_model()

    print(word2vec_model['ciao'])
    exit()
    # print('tuple_embedding with attributes_list: {0}'.format(attributes_list))
    # print('if [], all attributes are taken')
    # for index in tqdm(table.index):  # ITERATE over each row in the dataset
    #     embedding = doc2vec_embedding(table.loc[index], attributes_list)
    #     embeddings.append(embedding)

    return np.array(embeddings)
