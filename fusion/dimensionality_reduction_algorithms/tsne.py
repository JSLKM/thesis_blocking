from sklearn.manifold import TSNE

tsne_model = None

def set_tsne_model(num_components, perplexity, early_exaggeration, method, verbose):
    global tsne_model
    #Perplexity balances the attention t-SNE gives to local and 
    #global aspects of the data and can have large effects on the resulting plot.
    tsne_model = TSNE(
        perplexity=perplexity, 
        n_components=num_components, 
        method=method,
        early_exaggeration=early_exaggeration,
        init='pca', 
        n_iter=2500, 
        random_state=23
    )

def tsne_dim_reduction(embeddings, key_values):
    print("starting dimension: {0}".format(len(embeddings[0])))
    set_tsne_model(
        key_values['num_components'], 
        key_values['perplexity'], 
        key_values['early_exaggeration'], 
        key_values['method'],
        key_values['verbose']
        )

    if key_values['method'] == 'exact':
        print('tsne using exact method')
        from .pca import pca_dim_reduction
        embeddings = pca_dim_reduction(embeddings, key_values)
    return tsne_model.fit_transform(embeddings)

