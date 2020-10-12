import math
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

from .algorithm_utilities import clusterArray_to_blockDict

def dendrogramPlot(embeddings):
    plt.figure(figsize=(10, 7))
    plt.title("Dendograms")
    dend = shc.dendrogram(shc.linkage(embeddings, method='ward'))


def hierarchy_cluster(embeddings, key_values):
    # SET PARAMETERS
    tot_embeddings = len(embeddings)
    if tot_embeddings > 100:
        NUM_CLUSTERS = 10
    elif tot_embeddings < 10:
        NUM_CLUSTERS = 1
    else:
        # can be set to 0.1 by default
        float_value = tot_embeddings * key_values['num_clusters_rate']
        NUM_CLUSTERS = math.ceil(float_value)


    if key_values['verbose'] > 0:
        print("NUM_CLUSTERS {0}".format(NUM_CLUSTERS))
    # CLUSTERING
    cluster = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, affinity='euclidean', linkage='ward')
    assigned_clusters = cluster.fit_predict(embeddings)
    return clusterArray_to_blockDict(assigned_clusters)
