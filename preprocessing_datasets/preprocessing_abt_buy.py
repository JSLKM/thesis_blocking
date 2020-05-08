import pandas as pd
import os

def clean_abt_buy():
    dirname = os.path.dirname(__file__)
    path_dataset1 = os.path.join(dirname, '../source_datasets/Abt-Buy/Abt.csv')
    path_dataset2 = os.path.join(dirname, '../source_datasets/Abt-Buy/Buy.csv')
    path_dataset3 = os.path.join(dirname, '../source_datasets/Abt-Buy/abt_buy_perfectMapping.csv')

    table1 = pd.read_csv(path_dataset1,encoding="ISO-8859-1")
    table2 = pd.read_csv(path_dataset2,encoding="ISO-8859-1")
    table_match = pd.read_csv(path_dataset3,encoding="ISO-8859-1")

    table2.drop('manufacturer',axis=1,inplace=True)
    table3 = table1.append(table2, ignore_index=True)
    missing_values_replace = {"id": 'unk', "name": 'unk', 'description': 'unk', 'price': 'unk'}

    table3.fillna(value=missing_values_replace, inplace=True)

    pairs = set()

    for pair in zip(table_match['idAbt'], table_match['idBuy']):
        ordered_pair = tuple(sorted((table3.loc[table3['id'] == pair[0]].index[0], table3.loc[table3['id'] == pair[1]].index[0])))
        pairs.add(ordered_pair)


    table3.drop('id', axis=1, inplace=True)

    return table3, pairs

# table, pairs = clean_abt_buy()
# print(table)