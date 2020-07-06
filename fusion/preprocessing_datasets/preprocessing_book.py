#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:49:22 2020

@author: fabioazzalini
"""

import pandas as pd
import numpy as np
from .preprocessing_utilities import ValueUtils
from statistics import mean
import multiprocessing
from multiprocessing import Pool
import csv
import os

def clean_book():
     dirname = os.path.dirname(__file__)
     path = os.path.join(dirname, '../source_datasets/books/books_conflicts.csv')
     
     data = pd.read_csv(path, dtype='str')
     data = data[['ISBN_10', 'authors', 'title', 'big_cate', 'seller_link']]
     return data

     data.drop_duplicates(['ISBN_10', 'seller_link', 'big_cate'], keep="first", inplace=True)
     data['numberOfAuthors'] = data['authors'].map(lambda x: len(ValueUtils.split_values(x)))
     data['dirtyAuthor'] = data['authors'].map(lambda x: ValueUtils.split_values(x))
     data = data.explode('dirtyAuthor')
     data['oldAuthor'] = data['dirtyAuthor'].map(lambda x: ValueUtils.clean_value(x))

     data = data.groupby(['ISBN_10', 'seller_link', 'big_cate'], group_keys=False).agg(author=('authors', list), dirtyAuthor=('dirtyAuthor', list), oldAuthor=('oldAuthor', list),title=('title', 'first')).reset_index()
     data['oldAuthor'] = data['oldAuthor'].map(lambda x: ValueUtils.retain_only_values_with_alphabet(x))
     data['oldAuthor'] = data['oldAuthor'].map(lambda x: ValueUtils.retain_only_short_known_values(x))
     data['oldAuthor'] = data['oldAuthor'].map(lambda x: list(x))

     data = data.explode('oldAuthor')
     data = data.dropna()
     data = data.reset_index()
