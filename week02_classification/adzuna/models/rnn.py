import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from adzuna.models.base_salary_model import BaseSalaryModel


class RNNSalaryPredictor(BaseSalaryModel):
    def __init__(self, embedding_matrix, n_cat_features, title_max_len, desc_max_len, bidirectional=False, hid_size=64):
        super().__init__(embedding_matrix, n_cat_features, hid_size)
        self.title_max_len = title_max_len
        self.desc_max_len = desc_max_len
        lstm_coef = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix))

        self.title_lstm_forward = torch.nn.LSTM(100, 50, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.title_fc = torch.nn.Linear(lstm_coef * 50, hid_size)

        self.description_lstm_forward = torch.nn.LSTM(100, 50, num_layers=1, batch_first=True,
                                                      bidirectional=bidirectional)
        self.desc_fc = torch.nn.Linear(lstm_coef * 50, hid_size)

        self.final_fc_1 = torch.nn.Linear(self.hid_size, 1)
        self.final_fc_2 = torch.nn.Linear(self.title_max_len + self.desc_max_len + 1, 1)

    def forward(self, batch):
        title_data, desc_data, cat_data, N = self.unnest_batch(batch)
        title_len, desc_len = batch['Title_len'], batch['Desc_len']

        embedded_title = self.embedding(title_data)
        embedded_title_packed = pack_padded_sequence(embedded_title, title_len.to('cpu'), batch_first=True,
                                                     enforce_sorted=False)
        lstm_title_forward_packed, _ = self.title_lstm_forward(embedded_title_packed)
        lstm_title_forward, _ = pad_packed_sequence(lstm_title_forward_packed, batch_first=True,
                                                    total_length=self.title_max_len)
        title_processed = self.title_fc(lstm_title_forward)

        embedded_desc = self.embedding(desc_data)
        embedded_desc_packed = pack_padded_sequence(embedded_desc, desc_len.to('cpu'), batch_first=True,
                                                    enforce_sorted=False)
        lstm_desc_forward_packed, _ = self.description_lstm_forward(embedded_desc_packed)
        lstm_desc_forward, _ = pad_packed_sequence(lstm_desc_forward_packed, batch_first=True,
                                                   total_length=self.desc_max_len)
        desc_processed = self.desc_fc(lstm_desc_forward)

        cat_processed = self.cat_bn(self.cat_relu(self.cat_layer(cat_data))).view(N, 1, self.hid_size)

        concated = torch.cat((title_processed, desc_processed, cat_processed), dim=1)

        processed = self.final_fc_2(self.final_fc_1(concated).view(N, -1)).view(N, )

        return processed
