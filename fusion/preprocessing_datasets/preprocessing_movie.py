import pandas as pd
import numpy as np
import multiprocessing
import csv
import os

from statistics import mean
from multiprocessing import Pool

from .preprocessing_utilities import ValueUtils


def clean_movie():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../source_datasets/movie/movie_corrected_new_conflicts_nodm.txt')
    data = pd.read_csv(path, sep='\t', index_col=False)
    data['numberOfDirectors'] = data['director'].map(lambda x: len(ValueUtils.split_values_movies(x)))
    data['dirtyDirector'] = data['director'].map(lambda x: ValueUtils.split_values_movies(x))
    data = data.explode('dirtyDirector')

    data['newDirector'] = data['dirtyDirector'].map(lambda x: ValueUtils.clean_value(x))
    data = data.groupby(['movie_id', 'source', 'genre'], group_keys=False).agg(director=('director', list), dirtyDirector=('dirtyDirector', list), newDirector=('newDirector', list), title=('title', 'first'),year=('year', 'first')).reset_index()
    data['newDirector'] = data['newDirector'].map(lambda x: ValueUtils.retain_only_values_with_alphabet(x))
    data['newDirector'] = data['newDirector'].map(lambda x: ValueUtils.retain_only_short_known_values(x))
    data['newDirector'] = data['newDirector'].map(lambda x: list(x))
    data = data.explode('newDirector')
    data = data.dropna()
    # data['newDirector'] = ''
    data = data.reset_index()
    return data

def set_clean_movie():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../source_datasets/movie/movie_cleaned.csv')
    data = pd.read_csv(path, dtype='str')
    return data

def getTruthGiu2020():
    dirname = os.path.dirname(__file__)
    path = os.path.join(
        dirname, '../source_datasets/movie/movies_golden_giu2020.csv')
    data = pd.read_csv(path, dtype='str')
    return data.movie_id, data.directors