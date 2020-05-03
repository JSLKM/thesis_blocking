from sklearn.decomposition import PCA

pca_model = None

def set_pca_model(num_components):
    
    if verbose > 0:
        # print("setting TSNE with n_components: {0} & perplexity: {1}".format(num_components, perplexity))
        print("setting PCA with n_components: {0}".format(num_components))
    
    global pca_model
    pca_model = PCA(n_components=num_components, random_state=88)

def pca_dim_reduction(embeddings, **kwargs):
    print("starting dimension: {0}".format(len(embeddings[0])))
    set_pca_model(kwargs['num_components'] ,kwargs['verbose'])
    return pca_model.fit_transform(embeddings)

