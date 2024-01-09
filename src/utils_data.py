import pickle
import torch

def data_loading(dir_dataset):
    
    with open(dir_dataset, 'rb') as handle:
        dataset = pickle.load(handle)

    print('loading data at ', dir_dataset)
    return torch.Tensor(dataset['node_feature']), torch.Tensor(dataset['gt_inc'])


