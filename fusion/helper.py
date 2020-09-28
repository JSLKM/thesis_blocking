import numpy as np
from preprocessing_datasets.preprocessing_utilities import ValueUtils

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