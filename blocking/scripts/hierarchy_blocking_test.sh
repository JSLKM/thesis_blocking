#!/bin/bash

python3 ../blocking_starter.py \
	--dataset cora \
	--attributes_list author title \
	--cluster_method hierarchy \
	--num_clusters 25 \
	--dimension_reduction tsne \
	--num_components 2 \
	--perplexity 40 \
	--early_exaggeration 25 \
	--embedding_type inferSent \
	--model_type bilstm \
	--rnn_dim 2048 \
	--model_version 2 \
	| tee -a ./outputs/outputs$(date +'%Y_%m_%d').txt
	
