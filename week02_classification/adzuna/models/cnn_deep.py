import torch

from adzuna.models.base_salary_model import BaseSalaryModel


class CNNSalaryPredictor(BaseSalaryModel):
    def __init__(self, embedding_matrix, n_cat_features, hid_size=64, max_len=500):
        super().__init__(embedding_matrix, n_cat_features, hid_size)
        self.title_convolution_1 = torch.nn.Conv1d(100, hid_size, kernel_size=3, padding=1)
        self.title_convolution_2 = torch.nn.Conv1d(100, hid_size, kernel_size=5, padding=2)
        self.title_convolution_3 = torch.nn.Conv1d(100, hid_size, kernel_size=7, padding=3)
        self.title_bn = torch.nn.BatchNorm1d(max_len * 3)
        self.title_pooling = torch.nn.MaxPool1d(kernel_size=3 * max_len)
        self.title_dropout_layer = torch.nn.Dropout(p=0.3)

        self.description_convolution_1 = torch.nn.Conv1d(100, hid_size, kernel_size=3, padding=1)
        self.description_convolution_2 = torch.nn.Conv1d(100, hid_size, kernel_size=5, padding=2)
        self.description_convolution_3 = torch.nn.Conv1d(100, hid_size, kernel_size=7, padding=3)
        self.description_bn = torch.nn.BatchNorm1d(max_len * 3)
        self.description_pooling = torch.nn.MaxPool1d(kernel_size=3 * max_len)
        self.description_dropout_layer = torch.nn.Dropout(p=0.3)

        self.cat_dropout = torch.nn.Dropout(p=0.3)

        self.final_fc_1 = torch.nn.Linear(64, 1)
        self.final_fc_2 = torch.nn.Linear(3, 1)

    def forward(self, batch):
        title_data, desc_data, cat_data, N = self.unnest_batch(batch)

        embedded_title = self.embedding(title_data)
        embedded_title = torch.transpose(embedded_title, 1, 2)

        embedded_desc = self.embedding(desc_data)
        embedded_desc = torch.transpose(embedded_desc, 1, 2)

        title_conv_1 = self.title_convolution_1(embedded_title)
        title_conv_2 = self.title_convolution_2(embedded_title)
        title_conv_3 = self.title_convolution_3(embedded_title)
        print(title_conv_1.shape)
        print(title_conv_2.shape)
        print(title_conv_3.shape)

        title_convolution = torch.cat((title_conv_1, title_conv_2, title_conv_3), dim=2)

        title_processed = self.title_dropout_layer(self.title_pooling(title_convolution))

        desc_conv_1 = self.description_convolution_1(embedded_desc)
        desc_conv_2 = self.description_convolution_2(embedded_desc)
        desc_conv_3 = self.description_convolution_3(embedded_desc)

        desc_convolution = torch.cat((desc_conv_1, desc_conv_2, desc_conv_3), dim=2)

        desc_processed = self.description_dropout_layer(self.description_pooling(desc_convolution))
        cat_processed = self.cat_dropout(self.cat_bn(
            self.cat_relu(self.cat_layer(cat_data))).view(N, self.hid_size, 1))

        concated = torch.cat((title_processed, desc_processed, cat_processed), dim=2)
        concated = torch.transpose(concated, 1, 2)
        processed = self.final_fc_2(self.final_fc_1(concated).view(N, 3, )).view(N, )
        return processed