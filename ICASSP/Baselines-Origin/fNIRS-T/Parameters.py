def parameters():
    mode = 1
    if not mode:
        data_root = '../../Data/EF-MI-MA/Pre-MI'
    else:
        data_root = '../../Data/EF-MI-MA/Pre-MA'
    dataset = ['A', 'B', 'C']
    dataset_id = 1
    models = ['fNIRS-T', 'fNIRS-PreT']
    models_id = 0
    print(dataset[dataset_id], models[models_id])
    return data_root, dataset[dataset_id], models[models_id]