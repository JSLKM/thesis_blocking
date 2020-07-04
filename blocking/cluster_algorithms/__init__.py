import time


def _cluster_algorithm(embeddings, parameters, key_values):
    if parameters['cluster_method'] == 'kMean':
        from .kMeans_cluster import kMean_cluster
        return kMean_cluster(embeddings, key_values)

    elif parameters['cluster_method'] == 'hierarchy':
        from .hierarchy_cluster import hierarchy_cluster
        return hierarchy_cluster(embeddings, key_values)

    elif parameters['cluster_method'] == 'birch':
        from .birch_cluster import birch_cluster
        return birch_cluster(embeddings, key_values)

    elif parameters['cluster_method'] == 'fuzzy':
        from .fuzzy_cluster import fuzzy_cluster
        return fuzzy_cluster(embeddings, key_values)

    elif parameters['cluster_method'] == 'DBSCAN':
        from .DBScan_cluster import DBSCAN_cluster
        return DBSCAN_cluster(embeddings, key_values)

    raise NameError('Not cluster method found')


def cluster_algorithm(embeddings, parameters, key_values):
    start_time = time.time()
    blocks = _cluster_algorithm(embeddings, key_values, key_values)
    if key_values['verbose'] > 0:
        print("Blocking time is: {0}".format(time.time() - start_time))
    return blocks
