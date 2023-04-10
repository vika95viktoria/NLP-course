import torch

from adzuna.models.base_salary_model import BaseSalaryModel


class SalaryPredictor(BaseSalaryModel):
    def __init__(self, embedding_matrix, n_cat_features, title_max_len, desc_max_len, hid_size=64):
        super().__init__(embedding_matrix, n_cat_features, hid_size)
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix))
        self.title_convolution = torch.nn.Conv1d(100, hid_size, kernel_size=3, padding=1)
        self.title_pooling = torch.nn.MaxPool1d(kernel_size=title_max_len)

        self.description_convolution = torch.nn.Conv1d(100, hid_size, kernel_size=3, padding=1)
        self.description_pooling = torch.nn.MaxPool1d(kernel_size=desc_max_len)

        self.final_fc_1 = torch.nn.Linear(64, 1)
        self.final_fc_2 = torch.nn.Linear(3, 1)

    def forward(self, batch):
        title_data, desc_data, cat_data, N = self.unnest_batch(batch)

        embedded_title = self.embedding(title_data)
        embedded_title = torch.transpose(embedded_title, 1, 2)

        embedded_desc = self.embedding(desc_data)
        embedded_desc = torch.transpose(embedded_desc, 1, 2)

        title_processed = self.title_pooling(self.title_convolution(embedded_title))
        desc_processed = self.description_pooling(self.description_convolution(embedded_desc))
        cat_processed = self.cat_bn(self.cat_relu(self.cat_layer(cat_data))).view(N, self.hid_size, 1)

        concated = torch.cat((title_processed, desc_processed, cat_processed), dim=2)
        concated = torch.transpose(concated, 1, 2)
        processed = self.final_fc_2(self.final_fc_1(concated).view(N, 3, )).view(N, )
        return processed
