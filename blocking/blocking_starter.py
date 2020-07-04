#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:13:20 2020

@author: MicheleJin
"""
import argparse
import time

from embedding_algorithms import sentence_embedding
from dimensionality_reduction_algorithms import dimension_reduction_algorithms
from preprocessing_datasets import load_dataset
from cluster_algorithms import cluster_algorithm
from evaluation import calc_index



parser = argparse.ArgumentParser(description='Blocking Clustering')

parser.add_argument("--verbose", type=int, default='1',choices=[0, 1, 2], help="increase output verbosity")
parser.add_argument("--dataset", type=str, default='restaurant', help='dataset')
parser.add_argument("--cluster_method", type=str, default='kMean', help='kMean/hierarchy')
parser.add_argument("--num_clusters", type=int, default='10', help='used in KMean')
parser.add_argument("--distance_algorithm", type=str, default='cosine', help='cosine/euclidean')
parser.add_argument("--dimension_reduction", type=str, default="", help='tsne/pca')
parser.add_argument("--num_components", type=int, default='2', help='new dimension size')
parser.add_argument("--perplexity", type=int, default='40')
parser.add_argument("--early_exaggeration", type=int, default='12')
parser.add_argument("--method", type=str,default='barnes_hut', help='barnes_hut/exact')
parser.add_argument("--attributes_list", default=[], nargs='+')
parser.add_argument("--embedding_type", type=str, default='word2vec', help='word2vec/wiki2vec/doc2vec/inferSent')
parser.add_argument("--model_type", type=str, default='lstm', help="lstm/bilstm VALID FOR 'rnn' blocking_method ONLY")
parser.add_argument("--char_level", action='store_true', help="train char or word level")  # BY DEFAULT WE USE WORD-LEVEL
parser.add_argument("--model_version", type=int, default=2, help="1/2 model version for fasttext")
parser.add_argument("--rnn_dim", type=int, default=300, help="Dimension of the rnn to be used (300/1024/2048)")
parser.add_argument("--eps", type=int, default=0.5, help="")
parser.add_argument("--min_samples", type=int, default=5, help="")

params, _ = parser.parse_known_args()

key_values = {
    'dataset': params.dataset,
    'cluster_method': params.cluster_method,
    'verbose': params.verbose,
    'num_clusters': params.num_clusters,
    'distance_algorithm': params.distance_algorithm,
    'attributes_list': params.attributes_list,
    'embedding_type': params.embedding_type,
    'model_type': params.model_type,
    'char_level': params.char_level,
    'model_version': params.model_version,
    'rnn_dim': params.rnn_dim,
    'dimension_reduction': params.dimension_reduction,
    'num_components': params.num_components,
    'perplexity': params.perplexity,
    'early_exaggeration' : params.early_exaggeration,
    'method': params.method,
    'min_samples': params.min_samples,
    'eps': params.eps,
}

#################################################################################

prog_start = time.time()
# 1) LOAD and PREPROCESS the dataset
dataset_name, table, pairs = load_dataset(key_values)
# 2) DO the embedding
embeddings = sentence_embedding(table, key_values)
# 3) DO dimension reduction
embeddings = dimension_reduction_algorithms(embeddings, key_values)
# 4) DO the blocking
blocks = cluster_algorithm(embeddings,key_values, key_values)
# 5) EVALUATE the blocking by means of RR, PC, PQ, FM
calc_index(blocks,table,pairs)

print()
print("Total Execution time is: {0}".format(time.time() - prog_start))