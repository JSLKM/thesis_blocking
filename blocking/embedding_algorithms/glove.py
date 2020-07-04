import itertools
import functools
import nltk
import os
import io
import numpy as np
from tqdm import tqdm

glove_model = None


def load_word_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    print('load_word_vectors')
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


def tokenize_attribute(sentence):
    words = nltk.word_tokenize(sentence)
    return words


def set_glove_model():
    global glove_model
    dirname = os.path.dirname(__file__)
    path_glove_model = os.path.join(
        dirname, '../embedding_pretrained_models/glove.840B.300d/glove.840B.300d.txt')
    glove_model = load_word_vectors(path_glove_model)


def mean_embedding_word_level(row):
    tuple_value = []
    for attribute in row.index:  # These are the columns/attributes of each row
        attribute_value = row.loc[attribute]
        words = tokenize_attribute(str(attribute_value))
        words_embedding = []
        for word in words:
            words_embedding.append(glove_model[word] if word in glove_model.keys(
            ) else glove_model['unk'])
        attribute_embedding = np.nanmean(words_embedding, axis=0).tolist()
        tuple_value.append(attribute_embedding)
    # concatenate all sub-lists in embedding tuple
    tuple_embedded = functools.reduce(lambda x, y: x + y, tuple_value)
    return tuple_embedded


def tuple_glove_embedding(table, attributes_list):
    embeddings = []  # This is a list of lists,where each contained list is the tuple representation of the tuple
    set_glove_model()
    for index in tqdm(table.index):  # iterate over each row in the dataset
        # m*d-size vector, represents the entire tuple
        embedding = mean_embedding_word_level(table.loc[index])
        embeddings.append(embedding)
    return np.array(embeddings)
