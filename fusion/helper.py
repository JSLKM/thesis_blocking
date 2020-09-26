import numpy as np

def load_by_index(table_group_by_isbn, isbn_list, true_authors, index):
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

def get_author_candidates(list_authors, blocks, lengthNecessary):
    print("lengthNecessary: {0}".format(lengthNecessary))
    possibleAuthor = []
    indexSortedByLength = getSortedKeyByLength(blocks)
    for index in indexSortedByLength:
        block = blocks[index]
        if len(block) >= lengthNecessary:
            possibleAuthor.append(get_author_candidate_by_block(list_authors, block))
        else:
            break
    return possibleAuthor
        