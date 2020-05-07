import time

def _cluster_algorithm(embeddings, parameters, key_values):
    if parameters['cluster_method'] == 'kMean':
        from .kMeans_cluster_blocking import kMean_cluster_blocking
        return kMean_cluster_blocking(embeddings, key_values)
    elif parameters['cluster_method'] == 'hierarchy':
        from .hierarchy_cluster_blocking import hierarchy_cluster_blocking
        return hierarchy_cluster_blocking(embeddings, key_values)
    raise NameError('Not cluster method found')

def cluster_algorithm(embeddings, parameters, key_values):
    start_time = time.time()
    blocks = _cluster_algorithm(embeddings,key_values, key_values)
    if key_values['verbose'] > 0:
        print("Blocking time is: {0}".format(time.time() - start_time))
    return blocks