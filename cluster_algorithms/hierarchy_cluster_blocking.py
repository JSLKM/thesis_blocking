from .algorithm_utilities import clusterArray_to_blockDict
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc


def hierarchy_cluster_blocking(embeddings, key_values):

    # SET PARAMETERS
    NUM_CLUSTERS = key_values['num_clusters']

    # CLUSTERING
    print('clustering with NUM_CLUSTERS = {0}, '.format(NUM_CLUSTERS))
    cluster = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, affinity='euclidean', linkage='ward')
    assigned_clusters = cluster.fit_predict(embeddings)
    return clusterArray_to_blockDict(assigned_clusters)



# plt.figure(figsize=(10, 7))
# plt.title("Customer Dendograms")
# dend = shc.dendrogram(shc.linkage(data, method='ward'))


