import argparse
import time

from embedding_algorithms import sentence_embedding, set_embedding_model
from dimensionality_reduction_algorithms import dimension_reduction_algorithms
from preprocessing_datasets import load_dataset
from cluster_algorithms import cluster_algorithm
from helper import load_by_index, get_author_candidates, getFinalAuthors, filterGoldenTruth
from plot_tools import plotChart, plotCluster

parser = argparse.ArgumentParser(description='Fusion Clustering')
parser.add_argument("--verbose", type=int, default='1',choices=[0, 1, 2], help="increase output verbosity")
parser.add_argument("--dataset", type=str, default='restaurant', help='dataset')
parser.add_argument("--cluster_method", type=str, default='kMean', help='kMean/hierarchy')
params, _ = parser.parse_known_args()

key_values = {
    'verbose': params.verbose,
    'dataset': params.dataset,
    'cluster_method': params.cluster_method,
}

#################################################################################

prog_start = time.time()
# LOAD and PREPROCESS the dataset
dataset_name, tableGroupByISBN, list_ISBN_10, golden_true = load_dataset(key_values)
# RUN ALGORITHMS

# EVALUATE the blocking by means of RR, PC, PQ, FM
# calc_index(blocks,table,pairs)

print()
print("Total Execution time is: {0}".format(time.time() - prog_start))
