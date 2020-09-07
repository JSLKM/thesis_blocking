#!/bin/bash


python3 ../fusion_starter.py \
	--dataset clean_book \
	| tee -a ./outputs/outputs$(date +'%Y_%m_%d').txt
	
