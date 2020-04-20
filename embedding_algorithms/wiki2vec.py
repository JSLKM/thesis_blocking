from wikipedia2vec import Wikipedia2Vec

import numpy as np
import os
import time
import nltk
from tqdm import tqdm


wiki2vec_model = None


def set_wiki2vec_model():
    start_time = time.time()
    global wiki2vec_model
    dirname = os.path.dirname(__file__)
    path = "../embedding_pretrained_models/enwiki_20180420_300d/enwiki_20180420_300d.pkl"
    model_path = os.path.join(dirname, path)
    wiki2vec_model = Wikipedia2Vec.load(model_path)
    print("Model Setup time is: {0}".format(time.time() - start_time))


def wiki2vec_embedding(row, attributes_list):
    sentence_vector = []
    if attributes_list == []:
        attributes_list = row.index

    numw = 0
    for attribute in attributes_list:
        tokens = nltk.word_tokenize(row.loc[attribute])
        for token in tokens:
            try:  # Some word like "'s" doesn't exit
                if numw == 0:
                    sentence_vector = wiki2vec_model.get_word_vector(token)
                else:
                    sentence_vector = np.add(
                        sentence_vector, wiki2vec_model.get_word_vector(token))
                numw += 1
            except:
                pass
    return np.asarray(sentence_vector) / numw


def tuple_wiki2vec_embedding(table, attributes_list):
    # List of lists, each contained list is the embedding of the tuple
    embeddings = []
    set_wiki2vec_model()

    print('tuple_embedding with attributes_list: {0}'.format(attributes_list))
    print('if [], all attributes are taken')
    for index in tqdm(table.index):  # ITERATE over each row in the dataset
        embedding = wiki2vec_embedding(table.loc[index], attributes_list)
        embeddings.append(embedding)

    return np.array(embeddings)
