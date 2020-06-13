import pandas as pd

def clean_book():

    table = pd.read_csv('../source_datasets_fusion/Book/book.txt', sep="\t", names=["Source", "ISBN", "Title", "Author list"])
    attributes = table.columns.values.tolist()
    table.drop('Source', axis=1, inplace=True)

    # table['class'] = table.loc[:, 'class'].str.replace("'", '')

    # for attr in attributes:
    #     #table[attr] = table.loc[:, attr].str.replace('"', '')
    #     table[attr] = lrstrip(table, attr)
    #     setEmpty(table, attr)
    #     table[attr] = setLowercase(table, attr)

    # entities = createEntities(table)
    return table

#table = clean_book()
#print(table.iloc[0])
