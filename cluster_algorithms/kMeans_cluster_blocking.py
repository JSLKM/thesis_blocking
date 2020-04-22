from embedding_algorithms.wiki2vec import tuple_wiki2vec_embedding
from embedding_algorithms.word2vec import tuple_word2vec_embedding
from embedding_algorithms.doc2vec import tuple_doc2vec_embedding
from embedding_algorithms.glove import tuple_glove_embedding
from embedding_algorithms.fastText import tuple_fastText_embedding
from embedding_algorithms.inferSent import tuple_inferSent_embedding
from .algorithm_utilities import clusterArray_to_blockDict
from nltk.cluster import KMeansClusterer
import nltk

# nltk.download('punkt')


def kMean_cluster_blocking(table, key_values):

    if key_values['embedding_type'] == 'doc2vec':
        embeddings = tuple_doc2vec_embedding(
            table, key_values['attributes_list'])
    elif key_values['embedding_type'] == 'word2vec':
        embeddings = tuple_word2vec_embedding(
            table, key_values['attributes_list'])
    elif key_values['embedding_type'] == 'inferSent':
        embeddings = tuple_inferSent_embedding(
            table,
            model_type=key_values['model_type'],
            char_level=key_values['char_level'],
            model_version=key_values['model_version'],
            rnn_dim=key_values['rnn_dim'])
    elif key_values['embedding_type'] == 'glove':
        embeddings = tuple_glove_embedding(
            table, key_values['attributes_list'])
    elif key_values['embedding_type'] == 'fastText':
        embeddings = tuple_fastText_embedding(
            table, key_values['attributes_list'])
    elif key_values['embedding_type'] == 'wiki2vec':
        embeddings = tuple_wiki2vec_embedding(
            table, key_values['attributes_list'])

    # SET PARAMETERS
    NUM_CLUSTERS = key_values['num_clusters']
    distance_algorithm = nltk.cluster.util.cosine_distance
    if key_values['distance_algorithm'] == 'euclidean':
        distance_algorithm = nltk.cluster.util.euclidean_distance

    # CLUSTERING
    print('clustering with NUM_CLUSTERS = {0}, distance_algorithm = {1}'.format(
        NUM_CLUSTERS, distance_algorithm))
    kclusterer = KMeansClusterer(
        NUM_CLUSTERS, distance=distance_algorithm, avoid_empty_clusters=True, repeats=25)
    assigned_clusters = kclusterer.cluster(embeddings, assign_clusters=True)

    return clusterArray_to_blockDict(assigned_clusters)
