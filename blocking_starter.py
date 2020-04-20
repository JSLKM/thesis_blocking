#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:13:20 2020

@author: MicheleJin
"""

from preprocessing_blocking import load_dataset
from cluster_algorithms.kMeans_cluster_blocking import kMean_cluster_blocking
from evaluation import *

import argparse
import time

parser = argparse.ArgumentParser(description='Blocking Clustering')

parser.add_argument("--verbose", type=int, default='0',
                    choices=[0, 1, 2], help="increase output verbosity")
parser.add_argument("--dataset", type=str,
                    default='restaurant', help='dataset')
parser.add_argument("--cluster_method", type=str,
                    default='kMean', help='kMean')
parser.add_argument("--num_clusters", type=int,
                    default='10', help='used in KMean')
parser.add_argument("--distance_algorithm", type=str,
                    default='cosine', help='cosine/euclidean')
parser.add_argument("--attributes_list", default=[], nargs='+')
parser.add_argument("--embedding_type", type=str, default='word2vec',
                    help='word2vec/wiki2vec/doc2vec/inferSent')

params, _ = parser.parse_known_args()

parameters = {
    'dataset': params.dataset,
    'cluster_method': params.cluster_method,
}

key_values = {
    'verbose': params.verbose,
    'num_clusters': params.num_clusters,
    'distance_algorithm': params.distance_algorithm,
    'attributes_list': params.attributes_list,
    'embedding_type': params.embedding_type,
}

# print(key_values['attributes_list'])
# exit()

#################################################################################

# 1) LOAD and PREPROCESS the dataset

dataset_name, table, pairs = load_dataset(parameters['dataset'])
print("CURRENT dataset: "+dataset_name+"\n")

# 2) DO the blocking

start_time = time.time()

if parameters['cluster_method'] == 'kMean':
    if key_values['verbose'] == 2:
        print("Using Method: KMean")
    blocks = kMean_cluster_blocking(table, key_values)


# 3) EVALUATE the blocking by means of RR, PC, PQ, FM

compute_positive(pairs, blocks)
reduction_ratio = compute_reduction_ratio(table)
pair_completeness = compute_pair_completeness()
pair_quality = compute_pair_quality()
fmeasure = compute_fmeasure(pair_completeness, pair_quality)
reference_metric = 0 if (reduction_ratio == 0 or pair_completeness == 0) else (
    2*reduction_ratio*pair_completeness)/(reduction_ratio+pair_completeness)

end_time = time.time()
execution_time = end_time - start_time

print("(RR) Reduction ratio is: {0}".format(reduction_ratio))
print("(PC) Pair completeness is: {0}".format(pair_completeness))
print("(RM) Reference metric (Harmonic mean RR and PC) is: {0}".format(
    reference_metric))
print("(PQ) Pair quality is: {0}".format(pair_quality))
print("(FM) Fmeasure is: {0}".format(fmeasure))
print("(ET) Execution time is: {0}".format(time.time() - start_time))
