import numpy as np
from fuzzywuzzy import fuzz

from preprocessing_datasets.preprocessing_utilities import ValueUtils
from embedding_algorithms import sentence_embedding, set_embedding_model
from dimensionality_reduction_algorithms import dimension_reduction_algorithms
from preprocessing_datasets import load_dataset
from cluster_algorithms import cluster_algorithm
from plot_tools import plotChart, plotCluster

def launchWithReductionFusion(tableGroupByISBN, list_ISBN_10, golden_true, key_values):
    finalAuthors = []
    realAuthors = []
    for index in range(0,len(list_ISBN_10)):
        table_ISBN, list_authors, true_author = load_by_index(tableGroupByISBN, list_ISBN_10, golden_true, index, key_values['verbose'])
        embeddings_tokens = sentence_embedding(table_ISBN, key_values)
        reduction_embeddings = dimension_reduction_algorithms(embeddings_tokens, key_values)
        blocks = cluster_algorithm(reduction_embeddings, key_values)
        listCandidates = get_author_candidates(list_authors, blocks, key_values['block_length_thresold'] * len(reduction_embeddings), key_values['verbose'])
        finalAuthor = getFinalAuthors(listCandidates)
        realAuthor = filterGoldenTruth(true_author)
        if (key_values['verbose'] > 0):
            plotChart(list_authors, reduction_embeddings)
            plotCluster(blocks, list_authors, key_values['num_clusters'], reduction_embeddings)
            print(listCandidates)
            print("{0} VS true_author: {1}".format(finalAuthor, realAuthor))
        finalAuthors.append(finalAuthor)
        realAuthors.append(realAuthor)
    return finalAuthors, realAuthors

def launchWithoutReductionFusion(tableGroupByISBN, list_ISBN_10, golden_true, key_values):
    finalAuthors = []
    realAuthors = []
    for index in range(0,len(list_ISBN_10)):
        table_ISBN, list_authors, true_author = load_by_index(tableGroupByISBN, list_ISBN_10, golden_true, index, key_values['verbose'])
        embeddings_tokens = sentence_embedding(table_ISBN, key_values)
        blocks = cluster_algorithm(embeddings_tokens, key_values)
        listCandidates = get_author_candidates(list_authors, blocks, key_values['block_length_thresold'] * len(embeddings_tokens), key_values['verbose'])
        finalAuthor = getFinalAuthors(listCandidates)
        realAuthor = filterGoldenTruth(true_author)
        if (key_values['verbose'] > 0):
            print(listCandidates)
            print("{0} VS true_author: {1}".format(finalAuthor, realAuthor))
        finalAuthors.append(finalAuthor)
        realAuthors.append(realAuthor)
    return finalAuthors, realAuthors

def load_by_index(table_group_by_isbn, isbn_list, true_authors, index, verbose):
    if (verbose > 0):
        print('ISBN: {0}'.format(isbn_list[index]))
        print('true author: {0}'.format(true_authors[index]))
    table_ISBN = table_group_by_isbn.get_group(isbn_list[index])
    return table_ISBN, table_ISBN['author'].tolist(), true_authors[index]

def getSortedKeyByLength(blocks):
    return sorted(blocks, key=lambda k: len(blocks[k]), reverse=True)

def _get_longest_block(blocks):
    longest_len = max(map(len, blocks.values()))
    max_lens = [k for k, v in blocks.items() if len(v) == longest_len]
    return blocks[max_lens[0]]

def get_author_candidate_by_block(list_authors, chosenBlock):
    candidates = np.take(list_authors, chosenBlock)
    unique, counts = np.unique(candidates, return_counts=True)
    return dict(zip(unique, counts))

def get_author_candidates(list_authors, blocks, lengthNecessary, verbose):
    possibleAuthor = []
    discartedAuthor = []

    indexSortedByLength = getSortedKeyByLength(blocks)
    for index in indexSortedByLength:
        block = blocks[index]
        if len(block) >= lengthNecessary:
            possibleAuthor.append(get_author_candidate_by_block(list_authors, block))
        else:
            discartedAuthor.append(get_author_candidate_by_block(list_authors, block))
    if (verbose > 0):
        print("Discarted candidate: {0}".format(discartedAuthor))
        print("Possible candidate: {0}".format(possibleAuthor))
        print("lengthNecessary: {0}".format(lengthNecessary))
    return possibleAuthor

def getFinalAuthors(listCandidates):
    solution = []
    for candidate in listCandidates:
        solution.append(max(candidate, key=candidate.get))
    return solution

def filterGoldenTruth(authors):
    authors = ValueUtils.split_values(authors)
    return [ValueUtils.clean_value(x) for x in authors]

## AWARE consume the predAuthors array each time
def transformInBinary(predAuthors, trueAuthors, verbose):
    result = []
    predLength = len(predAuthors)
    trueLength = len(trueAuthors)
    if trueLength != predLength:
        raise Exception("size not compatible")
    for index in range(0, predLength):
        predCandidates = predAuthors[index]
        trueCandidates = trueAuthors[index]
        goodGuess = True
        for trueCandidate in trueCandidates:
            exist = False
            for predCandidate in predCandidates:
                rating = fuzz.ratio(trueCandidate, predCandidate)
                if verbose > 0:
                    print(">>>{0} - {1} - {2}<<<".format(trueCandidate, predCandidate, rating))
                if rating > 80:
                    predCandidates.remove(predCandidate)
                    exist = True
                    break
            if not exist:
                goodGuess = False
                if verbose > 0:
                    print(">>>>append 0")
                result.append(0)
                break
        if goodGuess:
            if verbose > 0:
                print(">>>>append 1")
            result.append(1)
    return result

