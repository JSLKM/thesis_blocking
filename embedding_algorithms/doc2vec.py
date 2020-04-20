import numpy as np
import gensim.models as g
import nltk
import os
import time
from tqdm import tqdm


doc2vec_model = None

def set_doc2vec_model():
    start_time = time.time()
    global doc2vec_model
    dirname = os.path.dirname(__file__)
    path = "../embedding_pretrained_models/enwiki_dbow/doc2vec.bin"
    model_path = os.path.join(dirname, path)
    doc2vec_model = g.Doc2Vec.load(model_path)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Model Setup time is: {0}".format(execution_time))


def doc2vec(doc):
    start_alpha = 0.01
    infer_epoch = 1000
    return doc2vec_model.infer_vector(doc_words=doc, alpha=start_alpha, steps=infer_epoch)

def tokenize_attribute(sentence):
    tokens = []
    for word in sentence:
        token = nltk.word_tokenize(word)
        tokens.append(token)
        
    return sum(tokens, [])

def doc2vec_embedding(row, attributes_list=[]):
    sentence = []
    if attributes_list == []:
        attributes_list = row.index

    for attribute in attributes_list:
        sentence.append(row.loc[attribute])
    tokens = tokenize_attribute(sentence)
    return doc2vec(tokens)


def tuple_doc2vec_embedding(table, attributes_list):
    # List of lists, each contained list is the embedding of the tuple
    embeddings = []
    set_doc2vec_model()

    print('tuple_embedding with attributes_list: {0}'.format(attributes_list))
    print('if [], all attributes are taken')
    for index in tqdm(table.index):  # ITERATE over each row in the dataset
        embedding = doc2vec_embedding(table.loc[index], attributes_list)
        embeddings.append(embedding)

    return np.array(embeddings)