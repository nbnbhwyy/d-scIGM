import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import scipy.io as sio
import torch
import pandas as pd
from scipy.sparse import csr_matrix

class CustomDataset(Dataset):

    def __init__(self,data_path, labelname, dataname, mode='train'):

        self.mode = mode  
        data_path = data_path
        data = pd.read_csv(data_path+dataname+"", sep=',', index_col=0)
        label = pd.read_csv(data_path+labelname,sep=',',index_col=0)        
        self.voc = list(data.columns)#data['voc']
        label_np = np.zeros(len(label))
        dicts_label_index = {}
        label_index = 0 
        for index, value in enumerate(label):
            if value not in dicts_label_index:
                label_index += 1
                dicts_label_index[value] = label_index

        for index, value in enumerate(label):
            label_np[index] =  dicts_label_index[value]

        if mode == 'train':
            self.data = csr_matrix(data).astype(np.float32)
            self.label = label_np
        elif mode == 'test':
            self.data = csr_matrix(data).astype(np.float32)
            self.label = label_np

    def __getitem__(self, index):
        try:
            bow = np.squeeze(self.data[index].toarray())
        except:
            bow = np.squeeze(self.data[index])
        return bow, np.squeeze(self.label[index])

    def __len__(self):
        return self.data.shape[0]

def dataloader(data_path,labelname, dataname='', mode='train', batch_size=500, shuffle=True, drop_last=False, num_workers = 4):
    dataset = CustomDataset(data_path = data_path,labelname = labelname, dataname=dataname, mode=mode)
    if mode == 'train':
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = num_workers), dataset.voc
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers), dataset.voc       

            



