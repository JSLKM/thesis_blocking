def load_dataset(key_values):
    
    if key_values['dataset'] == 'cora':
        from .preprocessing_cora import clean_cora
        table, pairs = clean_cora()
        
    elif key_values['dataset'] == 'restaurant':
        from .preprocessing_restaurant import clean_restaurant
        table, pairs = clean_restaurant()
    
    elif key_values['dataset'] == 'abt_buy':
        from .preprocessing_abt_buy import clean_abt_buy
        table, pairs = clean_abt_buy()
    
    elif key_values['dataset'] == 'amzn_gp':
        from .preprocessing_amzn_gp import clean_amzn_gp
        table, pairs = clean_amzn_gp()

    elif key_values['dataset'] == 'census':
        from .preprocessing_census import clean_census
        table, pairs = clean_census()

    elif key_values['dataset'] == 'dblp_acm':
        from .preprocessing_dblp_acm import clean_dblp_acm
        table, pairs = clean_dblp_acm()

    elif key_values['dataset'] == 'febrl_dirty':
        from .preprocessing_febrl_dirty import clean_febrl_dirty
        table, pairs = clean_febrl_dirty()

    if key_values['verbose'] > 0:
        print("#####################################################################")
        print("CURRENT dataset:        "+key_values['dataset'])
        print("CURRENT cluster_method: "+key_values['cluster_method'])
        print("CURRENT embedding_type: "+key_values['embedding_type'])
        print("#####################################################################")
    return key_values['dataset'], table, pairs

