#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:57:55 2020

@author: MicheleJin
"""


def load_dataset(dataset_name):
    
    if dataset_name == 'cora':
        from preprocessing_datasets.preprocessing_cora import clean_cora
        table, pairs = clean_cora()
        
    elif dataset_name == 'restaurant':
        from preprocessing_datasets.preprocessing_restaurant import clean_restaurant
        table, pairs = clean_restaurant()
        
    return dataset_name, table, pairs
