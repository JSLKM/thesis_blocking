def load_dataset(key_values):
    
    if key_values['dataset'] == 'cora':
        from .preprocessing_cora import clean_cora
        table, pairs = clean_cora()
        
    elif key_values['dataset'] == 'restaurant':
        from .preprocessing_restaurant import clean_restaurant
        table, pairs = clean_restaurant()
    
    if key_values['verbose'] > 0:
        print("#####################################################################")
        print("CURRENT dataset:        "+key_values['dataset'])
        print("CURRENT cluster_method: "+key_values['cluster_method'])
        print("CURRENT embedding_type: "+key_values['embedding_type'])
        print("#####################################################################")
    return key_values['dataset'], table, pairs

