import argparse
import time

# from embedding_algorithms import sentence_embedding
# from dimensionality_reduction_algorithms import dimension_reduction_algorithms
from preprocessing_datasets import load_dataset
# from cluster_algorithms import cluster_algorithm
# from evaluation import calc_index

parser = argparse.ArgumentParser(description='Fusion Clustering')

parser.add_argument("--verbose", type=int, default='1',
                    choices=[0, 1, 2], help="increase output verbosity")
parser.add_argument("--dataset", type=str,
                    default='restaurant', help='dataset')
parser.add_argument("--cluster_method", type=str,
                    default='kMean', help='kMean/hierarchy')

params, _ = parser.parse_known_args()

key_values = {
    'verbose': params.verbose,
    'dataset': params.dataset,
    'cluster_method': params.cluster_method,
}

#################################################################################

prog_start = time.time()
# 1) LOAD and PREPROCESS the dataset
dataset_name, table, list_ISBN_10, golden_true = load_dataset(key_values)

print(dataset_name)
print(table)
print(list_ISBN_10)
print(golden_true)
# # 2) DO the embedding
# embeddings = sentence_embedding(table, key_values)
# # 3) DO dimension reduction
# embeddings = dimension_reduction_algorithms(embeddings, key_values)
# # 4) DO the blocking
# blocks = cluster_algorithm(embeddings,key_values, key_values)
# # 5) EVALUATE the blocking by means of RR, PC, PQ, FM
# calc_index(blocks,table,pairs)

print()
print("Total Execution time is: {0}".format(time.time() - prog_start))
