import pandas as pd
import itertools
import re
import os
# First we load the .csv file as a pandas table
# Pandas assume the first row is the list of attributes names
def clean_febrl_dirty():

    dirname = os.path.dirname(__file__)
    source = 'dataset_10000.csv'
    path_dataset = os.path.join(dirname, '../source_datasets/febrl_dirty/'+source)
    table = pd.read_csv(path_dataset, sep=',', dtype=str)
    table = table.rename(columns={' given_name': 'given_name', ' surname': 'surname',' street_number': 'street_number', ' address_1': 'address1',' address_2': 'address2', ' suburb': 'suburb',' postcode': 'postcode', ' state': 'state',' date_of_birth': 'date_of_birth', ' age': 'age',' phone_number': 'phone_number', ' soc_sec_id': 'soc_sec_id'})
    attributes = table.columns.values.tolist()
    attributes = attributes[:-1]


    # remove blank leading and trailing chars
    for attr in attributes:
        table[attr] = table.loc[:, attr].str.lstrip().str.rstrip()
        table[attr].replace('', 'unk', inplace=True)

    # we set all string attributes to lowercase format

    for attr in attributes:
        table[attr] = table[attr].str.lower()

    entities_name = set()


    for reco_id in table['rec_id']:
        m = re.search('(.+?)org', reco_id)
        if m:
            entities_name.add(m.group(1))

    entities = dict()

    for ent in entities_name:
        entities[ent] = list(table[table['rec_id'].str.startswith(ent)].index)


    pairs = set()
    for val in entities:
        for x in itertools.combinations(entities[val], 2):
            ordered_pair = tuple(sorted(x))
            pairs.add(ordered_pair)

    table.drop('rec_id', axis=1, inplace=True)
    table.drop(' blocking_number', axis=1, inplace=True)

    return table,pairs

# table, pairs = clean_febrl_dirty()
# print(table)