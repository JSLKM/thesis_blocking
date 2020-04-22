#!/bin/bash

# python3 ../blocking_starter.py \
# 	--dataset restaurant \
# 	--cluster_method kMean \
# 	--num_clusters 25 \
# 	--embedding_type inferSent \
# 	--distance_algorithm cosine \
# 	--attributes_list citruname city

# python3 ../blocking_starter.py \
# 	--dataset cora \
# 	--cluster_method kMean \
# 	--num_clusters 25 \
# 	--embedding_type doc2vec \
# 	--distance_algorithm cosine \
# 	--attributes_list author title 


python3 ../blocking_starter.py \
	--dataset restaurant \
	--cluster_method kMean \
	--num_clusters 10 \
	--distance_algorithm cosine \
	--attributes_list citruname city \
	--embedding_type glove \
	--model_type bilstm \
	--char_level True \
	--rnn_dim 2048 \
	
