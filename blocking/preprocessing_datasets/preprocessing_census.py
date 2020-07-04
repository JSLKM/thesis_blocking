import pandas as pd
import itertools
import os

def clean_census():
    #ATTENTION! CENSUS ORIGINAL DATASETS ('A' and 'B' were not clean!)
    # First we load the .csv file as a pandas table
    # Pandas assume the first row is the list of attributes names

    dirname = os.path.dirname(__file__)
    path_dataset = os.path.join(dirname, '../source_datasets/census/census.csv')
    dtype_dic = {"last_name": str, "first_name": str, 'middle_name': str,'zip_code': str, 'street_address': str,'relation': str,'id':str}
    table = pd.read_csv(path_dataset, sep=',', dtype=dtype_dic)
    attributes = table.columns.values.tolist()
    missing_values_replace = {"first_name": 'unk', "last_name": 'unk', 'middle_name': 'unk', 'street_address': 'unk',
                              'zip_code': '0','relation':'unk'}
    table.fillna(value=missing_values_replace, inplace=True)

    table['zip_code'] = table['zip_code'].apply(lambda x: str(int(float(x))))
    # remove blank leading and trailing chars
    attributes.remove('id')
    attributes.remove('relation')
    for attr in attributes:
        table[attr] = table.loc[:, attr].str.lstrip().str.rstrip()
    # we set all string attributes to lowercase format
    for attr in attributes:
        table[attr] = table[attr].str.lower()

    entities = dict()

    for ent in set(table['id']):
        entities[ent] = list(table[table['id'] == ent].index)

    pairs = set()
    for val in entities:
        for x in itertools.combinations(entities[val], 2):
            ordered_pair = tuple(sorted(x))
            pairs.add(ordered_pair)


    table.drop('id', axis=1, inplace=True)
    table.drop('relation', axis=1, inplace=True)


    return table,pairs
