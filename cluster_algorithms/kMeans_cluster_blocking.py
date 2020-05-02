from .algorithm_utilities import clusterArray_to_blockDict
from nltk.cluster import KMeansClusterer
import nltk

# nltk.download('punkt')


def kMean_cluster_blocking(embeddings, key_values):

    # SET PARAMETERS
    NUM_CLUSTERS = key_values['num_clusters']
    distance_algorithm = nltk.cluster.util.cosine_distance
    if key_values['distance_algorithm'] == 'euclidean':
        distance_algorithm = nltk.cluster.util.euclidean_distance

    # CLUSTERING
    print('clustering with NUM_CLUSTERS = {0}, distance_algorithm = {1}'.format(NUM_CLUSTERS, distance_algorithm))
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=distance_algorithm, avoid_empty_clusters=True, repeats=25)
    assigned_clusters = kclusterer.cluster(embeddings, assign_clusters=True)

    return clusterArray_to_blockDict(assigned_clusters)
