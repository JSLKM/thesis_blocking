from .algorithm_utilities import clusterArray_to_blockDict
from sklearn.cluster import DBSCAN


def DBSCAN_cluster(embeddings, key_values):

    print('DBScan_cluster')
    eps = key_values['eps']
    min_samples = key_values['min_samples']
    # CLUSTERING
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    assigned_clusters = clustering.fit_predict(embeddings)
    return clusterArray_to_blockDict(assigned_clusters)
