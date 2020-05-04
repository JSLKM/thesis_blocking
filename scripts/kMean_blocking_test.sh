#!/bin/bash

# python3 ../blocking_starter.py \
# 	--dataset restaurant \
# 	--cluster_method kMean \
# 	--num_clusters 25 \
# 	--embedding_type inferSent \
# 	--distance_algorithm cosine \
# 	--attributes_list citruname city

python3 ../blocking_starter.py \
	--dataset cora \
	--attributes_list author title \
	--cluster_method kMean \
	--num_clusters 10 \
	--dimension_reduction tsne \
	--num_components 2 \
	--perplexity 50 \
	--distance_algorithm cosine \
	--embedding_type inferSent \
	--model_type bilstm \
	--rnn_dim 2048 \
	--model_version 2 \
	| tee -a ./outputs/outputs$(date +'%Y_%m_%d').txt


# python3 ../blocking_starter.py \
# 	--dataset restaurant \
# 	--attributes_list citruname city \
# 	--cluster_method kMean \
# 	--num_clusters 10 \
# 	--distance_algorithm cosine \
# 	--embedding_type inferSent \
# 	--model_type bilstm \
# 	--rnn_dim 2048 \
# 	--model_version 2 \
# 	>> ./outputs/outputs$(date +'%Y_%m_%d').txt
	
