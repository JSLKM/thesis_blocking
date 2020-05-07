import time

def _dimension_reduction_algorithms(embeddings, key_values):
    if key_values['dimension_reduction'] == 'tsne':
        from .tsne import tsne_dim_reduction
        return tsne_dim_reduction(
            embeddings, 
            num_components=key_values['num_components'],
            verbose=key_values['verbose'],
            perplexity=key_values['perplexity'],
            method=key_values['method'])
    elif key_values['dimension_reduction'] == 'pca':
        from .pca import pca_dim_reduction
        return pca_dim_reduction(
            embeddings, 
            num_components=key_values['num_components'],
            verbose=key_values['verbose'])

def dimension_reduction_algorithms(embeddings, key_values):
    if key_values['dimension_reduction'] != '':
        start_time = time.time()
        embeddings = _dimension_reduction_algorithms(embeddings, key_values)
        if key_values['verbose'] > 0:
            print("Dimension reduction time is: {0}".format(time.time() - start_time))
    return embeddings