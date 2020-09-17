def load_dataset(key_values):

    if key_values['dataset'] == 'book':
        from .preprocessing_book import clean_book
        from .preprocessing_book import set_clean_book, getMergedTruth
        table = clean_book()
        list_ISBN_10, mergedTruth = getMergedTruth()

    if key_values['dataset'] == 'clean_book':
        from .preprocessing_book import set_clean_book, getMergedTruth
        table = set_clean_book()
        list_ISBN_10, mergedTruth = getMergedTruth()

    if key_values['dataset'] == 'merged_book':
        from .preprocessing_book import set_merged_books, get_merged_books_truth
        table = set_merged_books()
        list_ISBN_10, mergedTruth = get_merged_books_truth()

    #if key_values['verbose'] > 0:
    #    print("#####################################################################")
    #    print("CURRENT dataset:        "+key_values['dataset'])
        # print("CURRENT cluster_method: "+key_values['cluster_method'])
        # print("CURRENT embedding_type: "+key_values['embedding_type'])
    #    print("#####################################################################")
    return key_values['dataset'], table, list_ISBN_10, mergedTruth
