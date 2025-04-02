import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

class BaseCLIPDataset(Dataset):
    """Base dataset class with common functionality"""
    def __init__(self):
        self.data = dict()
        self.common_init_fields()
        
    def common_init_fields(self):
        """Initialize common fields for all datasets"""
        self.data['sequence'] = []
        self.data['lang_goal'] = []
        self.data['pts'] = []
        self.data['clip_feats'] = []
        self.data['clip_sims'] = []
        self.data['actions'] = []

    def add(self, episode, lang_goal, pts, clip_feats, clip_sims, actions):
        """Base add method for common fields"""
        self.data['sequence'].append(episode)
        self.data['lang_goal'].append(lang_goal)
        self.data['pts'].append(pts)
        self.data['clip_feats'].append(clip_feats)
        self.data['clip_sims'].append(clip_sims)
        self.data['actions'].append(actions)

    def save(self, name):
        save_path = os.path.join(os.path.dirname(__file__), 'data', name)
        np.save(save_path, self.data)

    def load(self, path, sample_num=None):
        data_dict = np.load(path, allow_pickle=True, encoding='latin1')
        data = pd.DataFrame(data_dict.item())

        for field in self.data.keys():
            if field in data:
                self.data[field] = data[field][:sample_num] if sample_num else data[field]

    def load_and_merge(self, paths, save=False):
        merge_data = dict()
        # Note: encoding='latin1' for dict saved with py2 and loaded in py3
        data_dict_0 = np.load(paths[0], allow_pickle=True, encoding='latin1')        
        data_dict_0 = pd.DataFrame(data_dict_0.item())
        
        # merge multiple dataset
        for key, value in data_dict_0.items():
            for i in range(1, len(paths)):
                data_dict_i = np.load(paths[i], allow_pickle=True, encoding='latin1')
                data_dict_i = pd.DataFrame(data_dict_i.item())
                merge_data[key] = pd.concat([value, data_dict_i[key]], ignore_index=True)

        if save:
            # resave the data
            timestamp = time.time()
            timestamp_value = datetime.datetime.fromtimestamp(timestamp)
            name = 'train_' + timestamp_value.strftime('%Y_%m_%d_%H_%M_%S_') + str(len(merge_data['sequence'])) + '.npy'
            save_path = os.path.join(os.path.dirname(__file__), 'data', name)
            np.save(save_path, merge_data)
            
    def __len__(self):
        return len(self.data['sequence'])

    def __getitem__(self, idx):
        return [self.data[key][idx] for key in self.data.keys()]

    def get_splits(self, n_test=0.1, n_validate=0.1):
        test_size = round(n_test * len(self))
        validate_size = round(n_validate * len(self))
        train_size = len(self) - test_size - validate_size
        return random_split(self, [train_size, test_size, validate_size])

class CLIPActionDataset(BaseCLIPDataset):
    """Dataset for action tasks with additional fields"""
    def __init__(self):
        super().__init__()
        self.data.update({
            'action_idx': [],
            'reward': [],
            'step': [],
            'done': []
        })

    def add(self, episode, episode_steps, lang_goal, pts, clip_feats, clip_sims, 
            actions, action_idx, reward, done):
        super().add(episode, lang_goal, pts, clip_feats, clip_sims, actions)
        self.data['step'].append(episode_steps)
        self.data['action_idx'].append(action_idx)
        self.data['reward'].append(reward)
        self.data['done'].append(done)

    def __getitem__(self, idx):
        return [
            self.data['sequence'][idx],
            self.data['lang_goal'][idx],
            self.data['pts'][idx],
            self.data['clip_feats'][idx],
            self.data['clip_sims'][idx],
            self.data['actions'][idx],
            self.data['action_idx'][idx],
            self.data['done'][idx]
        ]

class CLIPMultiPlaceDataset(BaseCLIPDataset):
    """Dataset for place testing tasks"""
    def __init__(self):
        super().__init__()
        self.data['gt_action_idxs'] = []

    def add(self, episode, lang_goal, pts, clip_feats, clip_sims, actions, gt_action_idxs):
        super().add(episode, lang_goal, pts, clip_feats, clip_sims, actions)
        self.data['gt_action_idxs'].append(gt_action_idxs)

    def __getitem__(self, idx):
        return [
            self.data['sequence'][idx],
            self.data['lang_goal'][idx],
            self.data['pts'][idx],
            self.data['clip_feats'][idx],
            self.data['clip_sims'][idx],
            self.data['actions'][idx],
            self.data['gt_action_idxs'][idx]
        ]