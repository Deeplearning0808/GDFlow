import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

import pandas as pd
import numpy as np
import random
import os

import controldiffeq

def load_SMD(machine_id, batch_size, window_size, stride_size, train_split):
    base_dir = "./benchmarks/SMD"
    train_df = pd.read_csv(os.path.join(base_dir, "train", f"{machine_id}.txt"), header=None)
    test_df = pd.read_csv(os.path.join(base_dir, "test", f"{machine_id}.txt"), header=None)
    test_label_df = pd.read_csv(os.path.join(base_dir, "test_label", f"{machine_id}.txt"), header=None)

    train_X = train_df.to_numpy()
    T, C = train_X.shape
    train_y = np.zeros((T,), dtype=int)
    test_X = test_df.to_numpy()
    test_y = test_label_df.to_numpy().squeeze(1)

    train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
    train_y, test_y = train_y.astype(int), test_y.astype(int)

    assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
    assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0

    scaler = StandardScaler()
    norm_train_feature = scaler.fit_transform(train_X) # (X,N)
    norm_test_feature = scaler.transform(test_X)
    
    entity_mean = np.mean(norm_train_feature, axis=0) # (N,)
    entity_covar = kneighbors_graph(X=norm_train_feature.T, n_neighbors=2, mode='distance', metric='cosine')
    entity_covar = np.array(entity_covar.todense(), dtype=np.float32)
    
    n_sensor = norm_train_feature.shape[1]
    split_len = int(train_split*norm_train_feature.shape[0])
    trainset = norm_train_feature[:split_len]
    valset = norm_train_feature[split_len:]
    print('trainset size',trainset.shape, 'valset size', valset.shape)
    
    testset = norm_test_feature
    test_label = test_y
    print('testset size',testset.shape, 'test_label size',test_label.shape, 'anomaly ratio', sum(test_label)/len(test_label))    

    train_dataset = SMD_dataset(trainset, window_size, stride_size)
    if train_split != 1.0:
        val_dataset = SMD_dataset(valset, window_size, stride_size)
    test_dataset = SMD_dataset(testset, window_size, stride_size, test_label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if train_split != 1.0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, n_sensor, entity_mean, entity_covar

class SMD_dataset(Dataset):
    def __init__(self, df, window_size, stride_size, labels=None) -> None:
        super(SMD_dataset, self).__init__()
        self.df = df
        self.window_size = window_size # 5
        self.stride_size = stride_size # 5
        if labels is None:
            labels = np.zeros(df.shape[0])
        self.data, self.label = self.preprocess(df, labels)
        self.times = torch.linspace(0, self.window_size-1, self.window_size)
        self.data = torch.FloatTensor(self.data)
        self.coeffs = controldiffeq.natural_cubic_spline_coeffs(self.times, self.data.transpose(1,2))
        
    def preprocess(self, df, labels):
        final_data = []
        final_label = []

        start_idx = np.arange(0, len(df) - self.window_size + 1, self.stride_size)
        for i in start_idx:
            temp_data = df[i:i + self.window_size, :]
            final_data.append(temp_data)
            if labels is not None:
                temp_label = labels[i + self.window_size - 1] # Using the label of lastest window
                final_label.append(temp_label)

        final_data = np.array(final_data)[..., np.newaxis] # (b,w,N,1)
        if labels is not None:
            final_label = np.array(final_label)  # (b,)
        else:
            final_label = None
        
        return final_data, final_label

    def __len__(self):
        length = len(self.label)
        return length

    def __getitem__(self, index):
        return self.data[index].transpose(0,1), self.coeffs[0][index], self.coeffs[1][index], self.coeffs[2][index], self.coeffs[3][index], self.label[index] # [b,N,w,1], (b,)
    