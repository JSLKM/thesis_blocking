import time

def _dimension_reduction_algorithms(embeddings, key_values):
    if key_values['dimension_reduction'] == 'tsne':
        from .tsne import tsne_dim_reduction
        if key_values['verbose'] > 0:
            print('dimension_reduction: {0}'.format(key_values['dimension_reduction']))
            print('num_components: {0}'.format(key_values['num_components']))
            print('perplexity: {0}'.format(key_values['perplexity']))
            print('early_exaggeration: {0}'.format(key_values['early_exaggeration']))
            print('method: {0}'.format(key_values['method']))
        return tsne_dim_reduction(embeddings, key_values)
    elif key_values['dimension_reduction'] == 'pca':
        from .pca import pca_dim_reduction
        if key_values['verbose'] > 0:
            print('dimension_reduction: {0}'.format(key_values['dimension_reduction']))
            print('num_components: {0}'.format(key_values['num_components']))
        return pca_dim_reduction(embeddings, key_values)

def dimension_reduction_algorithms(embeddings, key_values):
    if key_values['dimension_reduction'] != '':
        start_time = time.time()
        embeddings = _dimension_reduction_algorithms(embeddings, key_values)
        if key_values['verbose'] > 0:
            print("Dimension reduction time is: {0}".format(time.time() - start_time))
    return embeddings