import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

from .algorithm_utilities import clusterArray_to_blockDict


def fuzzy_cluster(embeddings, key_values):

    # SET PARAMETERS
    NUM_CLUSTERS = key_values['num_clusters']

    # CLUSTERING
    print('clustering with NUM_CLUSTERS = {0}, '.format(NUM_CLUSTERS))
    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
        embeddings, c=NUM_CLUSTERS, m=2, error=0.005, maxiter=1000, seed=25)

    print(u_orig)

    # Show 3-cluster model
    fig2, ax2 = plt.subplots()
    ax2.set_title('Trained model')
    for j in range(NUM_CLUSTERS):
        ax2.plot(embeddings[0, u_orig.argmax(axis=0) == j],
                 embeddings[1, u_orig.argmax(axis=0) == j], 'o',
                 label='series ' + str(j))
    ax2.legend()
    plt.show()
    exit()

