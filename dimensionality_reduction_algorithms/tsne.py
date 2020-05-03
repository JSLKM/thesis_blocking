from sklearn.manifold import TSNE

tsne_model = None

def set_tsne_model(num_components, perplexity, method, verbose):
    
    if verbose > 0:
        print("setting TSNE with n_components: {0} & perplexity: {1}".format(num_components, perplexity))

    global tsne_model
    #Perplexity balances the attention t-SNE gives to local and 
    #global aspects of the data and can have large effects on the resulting plot.
    tsne_model = TSNE(
        perplexity=perplexity, 
        n_components=num_components, 
        method=method,
        init='pca', 
        n_iter=2500, 
        random_state=23
    )

def tsne_dim_reduction(embeddings, **kwargs):
    print("starting dimension: {0}".format(len(embeddings[0])))
    set_tsne_model(kwargs['num_components'], kwargs['perplexity'], kwargs['method'],kwargs['verbose'])
    return tsne_model.fit_transform(embeddings)

