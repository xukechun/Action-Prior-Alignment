from torch.utils.data import DataLoader
from helpers.datasets import CLIPActionDataset, CLIPMultiPlaceDataset

def data_merge(dataset, paths):
    dataset.load_and_merge(paths)    
    return 

def get_data_loader(dataset_class, path, sample_num=None, shuffle=False):
    dataset = dataset_class()
    dataset.load(path, sample_num)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle)

def unified_data_loader(path, sample_num=None, shuffle=False):
    return get_data_loader(CLIPActionDataset, path, sample_num, shuffle)

def unified_adaptive_data_loader(path, sample_num=None, shuffle=False):
    return get_data_loader(CLIPMultiPlaceDataset, path, sample_num, shuffle)
