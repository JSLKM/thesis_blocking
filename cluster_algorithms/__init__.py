
def cluster_algorithm(embeddings, parameters, key_values):
    if parameters['cluster_method'] == 'kMean':
        from .kMeans_cluster_blocking import kMean_cluster_blocking
        return kMean_cluster_blocking(embeddings, key_values)
    return None