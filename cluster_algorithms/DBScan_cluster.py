from .algorithm_utilities import clusterArray_to_blockDict
from sklearn.cluster import DBSCAN


def DBSCAN_cluster(embeddings, key_values):

    # CLUSTERING
    clustering = DBSCAN(eps=3, min_samples=5)
    assigned_clusters = clustering.fit_predict(embeddings)
    return clusterArray_to_blockDict(assigned_clusters)
