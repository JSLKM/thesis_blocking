import numpy as np
import itertools

found_true_positives = set()
all_found_pairs = set()
true_positives = set()

# Measure of Test accuracy
def compute_fmeasure(pair_completeness, pair_quality):
	if pair_completeness == 0:
		return 0.0
	return 2 * (pair_completeness*pair_quality) / (pair_completeness + pair_quality)

def compute_pair_quality():
	global found_true_positives
	global all_found_pairs
	pair_quality = 0
	if len(all_found_pairs) > 0:
		pair_quality = len(found_true_positives) / len(all_found_pairs)
	return pair_quality
	
# Measures how much the blocking technique can reduce
# the number of pair comparisons with respect to 
# a naive approach
def compute_reduction_ratio(table):
	global all_found_pairs
	num_rows = len(table.index)
	total_possible_pairs = int((num_rows * (num_rows - 1)) / 2)
	reduction_ratio = 1.0 - len(all_found_pairs) / total_possible_pairs
	return reduction_ratio

# Measures the effectiveness of the blocking method
# at not removing true matches from the set of comparisons.
def compute_pair_completeness():
	global true_positives
	global found_true_positives
	return len(found_true_positives) / len(true_positives)

def compute_positive(pairs, blocks):
	global all_found_pairs
	global true_positives
	global found_true_positives

	# SETUP 
	if len(all_found_pairs) > 0 or len(found_true_positives) > 0:
		all_found_pairs.clear()
		found_true_positives.clear()

	true_positives = pairs

	for block in blocks.keys():
		for record_pair in tuple(itertools.combinations(blocks[block], 2)):
			ordered_pair = tuple(sorted(record_pair))
			if ordered_pair not in all_found_pairs:
				all_found_pairs.add(ordered_pair)
				if ordered_pair in true_positives:
					found_true_positives.add(ordered_pair)

def calc_index(blocks, table, pairs):
	compute_positive(pairs, blocks)
	reduction_ratio = compute_reduction_ratio(table)
	pair_completeness = compute_pair_completeness()
	pair_quality = compute_pair_quality()
	fmeasure = compute_fmeasure(pair_completeness, pair_quality)
	reference_metric = 0 if (reduction_ratio == 0 or pair_completeness == 0) else (2*reduction_ratio*pair_completeness)/(reduction_ratio+pair_completeness)

	print("(RR) Reduction ratio is: {0}".format(reduction_ratio))
	print("(PC) Pair completeness is: {0}".format(pair_completeness))
	print("(RM) Reference metric (Harmonic mean RR and PC) is: {0}".format(reference_metric))
	print("(PQ) Pair quality - Precision is: {0}".format(pair_quality))
	print("(FM) Fmeasure is: {0}".format(fmeasure))
	

