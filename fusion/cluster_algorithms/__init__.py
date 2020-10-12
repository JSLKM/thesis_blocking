import time


def _cluster_algorithm(embeddings, key_values):
    if key_values['cluster_method'] == 'hierarchy':
        from .hierarchy_cluster import hierarchy_cluster
        if key_values['verbose'] > 0:
            print('cluster_method: {0}'.format(key_values['cluster_method']))
            print('num_clusters_rate: {0}'.format(key_values['num_clusters_rate']))
        return hierarchy_cluster(embeddings, key_values)

    elif key_values['cluster_method'] == 'DBSCAN':
        from .DBScan_cluster import DBSCAN_cluster
        if key_values['verbose'] > 0:
            print('cluster_method: {0}'.format(key_values['cluster_method']))
            print('eps: {0}'.format(key_values['eps']))
            print('min_samples: {0}'.format(key_values['min_samples']))
        return DBSCAN_cluster(embeddings, key_values)

    raise NameError('Not cluster method found')


def cluster_algorithm(embeddings, key_values):
    start_time = time.time()
    blocks = _cluster_algorithm(embeddings, key_values)
    if key_values['verbose'] > 0:
        print("Blocking time is: {0}".format(time.time() - start_time))
    return blocks
