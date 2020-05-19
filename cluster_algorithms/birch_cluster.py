from .algorithm_utilities import clusterArray_to_blockDict
from sklearn.cluster import Birch


def birch_cluster(embeddings, key_values):

    # SET PARAMETERS
    NUM_CLUSTERS = key_values['num_clusters']

    # CLUSTERING
    print('clustering with NUM_CLUSTERS = {0}, '.format(NUM_CLUSTERS))
    brc = Birch(n_clusters=NUM_CLUSTERS)
    assigned_clusters = brc.fit_predict(embeddings)
    return clusterArray_to_blockDict(assigned_clusters)
