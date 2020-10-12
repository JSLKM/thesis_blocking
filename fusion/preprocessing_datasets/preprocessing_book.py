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

book_db = None
ISBN_10_groups = None

def clean_book():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../source_datasets/books/books_conflicts.csv')
    data = pd.read_csv(path, dtype='str')
    data = data[['ISBN_10', 'authors', 'title', 'big_cate', 'seller_link']]
    data.drop_duplicates(['ISBN_10', 'seller_link', 'big_cate'], keep="first", inplace=True)
    data['numberOfAuthors'] = data['authors'].map(lambda x: len(ValueUtils.split_values(x)))
    data['dirtyAuthor'] = data['authors'].map(lambda x: ValueUtils.split_values(x))
    data = data.explode('dirtyAuthor')
    data['author'] = data['dirtyAuthor'].map(lambda x: ValueUtils.clean_value(x))
    data = data.groupby(['ISBN_10', 'seller_link', 'big_cate'], group_keys=False).agg(authors=('authors', list), dirtyAuthor=('dirtyAuthor', list), author=('author', list), title=('title', 'first')).reset_index()
    data['author'] = data['author'].map(lambda x: ValueUtils.retain_only_values_with_alphabet(x))
    data['author'] = data['author'].map(lambda x: ValueUtils.retain_only_short_known_values(x))
    data['author'] = data['author'].map(lambda x: list(x))
    data = data.explode('author')
    data = data.dropna()
    data = data.reset_index()
    return data

def load_groupby_ISBN(isbn):
    book_db = set_clean_book()
    ISBN_10_groups = book_db.groupby('ISBN_10')
    return ISBN_10_groups.get_group(isbn)

def load_book_by_path(givenPath):
    # givenPath = '../source_datasets/book/book_detail_fiction_childrens_fiction_young_adult.txt'
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, givenPath)
    data = pd.read_csv(path, dtype='str')
    return data

# Truth
def getTruthMag2020():
    dirname = os.path.dirname(__file__)
    path = os.path.join(
        dirname, '../source_datasets/books/books_golden_mag2020.csv')
    data = pd.read_csv(path, dtype='str')
    return data.ISBN_10, data.true_authors

def getTruthGiu2020():
    dirname = os.path.dirname(__file__)
    path = os.path.join(
        dirname, '../source_datasets/books/books_golden_giu2020.csv')
    data = pd.read_csv(path, dtype='str')
    return data.ISBN_10, data.true_authors

def getTruth500():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../source_datasets/books/truth500.csv')
    data = pd.read_csv(path, dtype='str', engine='python')
    return data.ISBN_10, data.true_authors

def getMerged():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../source_datasets/books/merged_truth.csv')
    data = pd.read_csv(path, dtype='str', engine='python')
    return data.ISBN_10, data.true_authors

def returnMergedTruth():
    dirname = os.path.dirname(__file__)
    path1 = os.path.join(dirname, '../source_datasets/books/books_golden_mag2020.csv')
    data1 = pd.read_csv(path1, dtype='str')
    path2 = os.path.join(dirname, '../source_datasets/books/books_golden_giu2020.csv')
    data2 = pd.read_csv(path2, dtype='str')
    path3 = os.path.join(dirname, '../source_datasets/books/truth500.csv')
    data3 = pd.read_csv(path2, dtype='str', engine='python')
    mergedData = pd.concat([data1, data2]).drop_duplicates(subset=['ISBN_10']).reset_index(drop=True)
    mergedData = pd.concat([mergedData, data3]).drop_duplicates(subset=['ISBN_10']).reset_index(drop=True)
    return mergedData


# Data source
def set_clean_book():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../source_datasets/books/books_cleaned.csv')
    data = pd.read_csv(path, dtype='str')
    return data

def set_merged_books(): 
    dirname = os.path.dirname(__file__)
    #path = os.path.join(dirname, '../source_datasets/books/books_merged.csv')
    path = os.path.join(dirname, '../source_datasets/books/books_merged_cleaned.csv')
    data = pd.read_csv(path, dtype='str')
    return data


# Experiment with subset of books
def get_merged_books_truth():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../source_datasets/books/mergedBookTruth20.csv')
    data = pd.read_csv(path, dtype='str', engine='python')
    return data.ISBN_10, data.true_authors

def get_merged_books_multi_authors_truth():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../source_datasets/books/multipleAuthors.csv')
    data = pd.read_csv(path, dtype='str', engine='python')
    return data.ISBN_10, data.true_authors