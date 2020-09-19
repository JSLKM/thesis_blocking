from sklearn.decomposition import PCA

pca_model = None

def set_pca_model(num_components, verbose):
    global pca_model
    pca_model = PCA(n_components=num_components, random_state=88)

def pca_dim_reduction(embeddings, key_values):
    # print("starting dimension: {0}".format(len(embeddings[0])))
    set_pca_model(key_values['num_components'] ,key_values['verbose'])
    return pca_model.fit_transform(embeddings)

