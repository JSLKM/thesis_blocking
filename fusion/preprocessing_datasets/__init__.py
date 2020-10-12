import time

def load_dataset(key_values):
    start_time = time.time()

    if key_values['dataset'] == 'clean_movie':
        from .preprocessing_movie import set_clean_movie, getTruthGiu2020
        table = set_clean_movie()
        listIds, truthDirector = getTruthGiu2020()
        if key_values['verbose'] > 0:
            print("dataset: {0}".format(key_values['dataset']))
            print("Loading time is: {0}".format(time.time() - start_time))
        return key_values['dataset'],  table.groupby('movie_id'), listIds, truthDirector

    if key_values['dataset'] == 'book':
        from .preprocessing_book import clean_book
        from .preprocessing_book import getMerged
        table = clean_book()
        list_ISBN_10, mergedTruth = getMerged()

    if key_values['dataset'] == 'clean_book':
        from .preprocessing_book import set_clean_book, getMerged
        table = set_clean_book()
        list_ISBN_10, mergedTruth = getMerged()

    ## Get subset of books
    if key_values['dataset'] == 'merged_book':
        from .preprocessing_book import set_merged_books, get_merged_books_truth
        table = set_merged_books()
        list_ISBN_10, mergedTruth = get_merged_books_truth()
    
    if key_values['dataset'] == 'merged_book-multiAuthors':
        from .preprocessing_book import set_merged_books, get_merged_books_multi_authors_truth
        table = set_merged_books()
        list_ISBN_10, mergedTruth = get_merged_books_multi_authors_truth()

    if key_values['verbose'] > 0:
        print("dataset: {0}".format(key_values['dataset']))
        print("Loading time is: {0}".format(time.time() - start_time))
    return key_values['dataset'], table.groupby('ISBN_10'), list_ISBN_10, mergedTruth
