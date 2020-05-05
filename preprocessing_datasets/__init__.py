def load_dataset(dataset_name):
    
    if dataset_name == 'cora':
        from .preprocessing_cora import clean_cora
        table, pairs = clean_cora()
        
    elif dataset_name == 'restaurant':
        from .preprocessing_restaurant import clean_restaurant
        table, pairs = clean_restaurant()
        
    return dataset_name, table, pairs