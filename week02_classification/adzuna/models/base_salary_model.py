import torch.nn as nn
import torch
from abc import abstractmethod


class BaseSalaryModel(nn.Module):

    def __init__(self, embedding_matrix, n_cat_features, hid_size=64):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix))

        self.cat_layer = torch.nn.Linear(n_cat_features, hid_size)
        self.cat_relu = torch.nn.ReLU()
        self.cat_bn = torch.nn.BatchNorm1d(hid_size)

        self.hid_size = hid_size

    def unnest_batch(self, batch):
        title_data = batch['Title']
        desc_data = batch['FullDescription']
        cat_data = batch['Categorical']
        N = title_data.shape[0]
        return title_data, desc_data, cat_data, N

    @abstractmethod
    def forward(self, batch):
        pass
