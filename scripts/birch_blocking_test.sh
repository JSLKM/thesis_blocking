#!/bin/bash

python3 ../blocking_starter.py \
	--dataset cora \
	--attributes_list author title \
	--cluster_method birch \
	--num_clusters 5 \
	--dimension_reduction tsne \
	--num_components 2 \
	--perplexity 40 \
	--embedding_type inferSent \
	--model_type bilstm \
	--rnn_dim 2048 \
	--model_version 2 \
	| tee -a ./outputs/outputs$(date +'%Y_%m_%d').txt
	
