def load_by_index(table_group_by_isbn, isbn_list, true_authors, index):
    print('ISBN: {0}'.format(isbn_list[index]))
    print('true author: {0}'.format(true_authors[index]))
    table_ISBN = table_group_by_isbn.get_group(isbn_list[index])
    return table_ISBN, table_ISBN['authors'].tolist(), true_authors[index]