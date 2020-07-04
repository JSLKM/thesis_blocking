import fasttext
import functools
import nltk
import os
import numpy as np
import time
from tqdm import tqdm

fastText_model = None


def tokenize_attribute(sentence):
    words = nltk.word_tokenize(sentence)
    return words


def set_fastText_model():
    start_time = time.time()
    global fastText_model
    dirname = os.path.dirname(__file__)
    path_fasttext_model = os.path.join(
        dirname, '../embedding_pretrained_models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin')
    fastText_model = fasttext.FastText.load_model(path_fasttext_model)
    print("Model Setup time is: {0}".format(time.time() - start_time))


def mean_embedding_char_level(row):
    tuple_value = []
    for attribute in row.index:  # These are the columns/attributes of each row
        attribute_value = row.loc[attribute]
        # print(str(attribute_value))
        words = tokenize_attribute(str(attribute_value))
        # print(words)
        words_embedding = []
        for word in words:
            word_vect = fastText_model.get_word_vector(word)
            words_embedding.append(word_vect)

        attribute_embedding = np.nanmean(words_embedding, axis=0).tolist()
        # print('---------------------------')
        # print(attribute_embedding)
        # print(type(attribute_embedding))
        tuple_value.append(attribute_embedding)
    # concatenate all sub-lists in embedding tuple
    # print(type(tuple_value[0][0]))
    # print(len(tuple_value))
    # print(tuple_value)
    tuple_embedded = functools.reduce(lambda x, y: x + y, tuple_value)

    return tuple_embedded


def tuple_fastText_embedding(table, attributes_list):
    embeddings = []  # This is a list of lists,where each contained list is the tuple representation of the tuple
    set_fastText_model()

    for index in tqdm(table.index):  # iterate over each row in the dataset
        # m*d-size vector, represents the entire tuple
        embedding = mean_embedding_char_level(table.loc[index])
        embeddings.append(embedding)
    return np.array(embeddings)
