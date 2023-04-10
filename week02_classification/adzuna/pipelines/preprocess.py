import numpy as np
import torch
from adzuna.config import *


class BatchPreprocessor:
    def __init__(self, token_to_id, categorical_vectorizer, device, title_max_len, desc_max_len):
        self.token_to_id = token_to_id
        self.categorical_vectorizer = categorical_vectorizer
        self.device = device
        self.title_max_len = title_max_len
        self.desc_max_len = desc_max_len

    @staticmethod
    def apply_word_dropout(matrix, keep_prop):
        dropout_mask = np.random.choice(2, np.shape(matrix), p=[keep_prop, 1 - keep_prop])
        dropout_mask &= matrix != PAD_IX
        return np.choose(dropout_mask, [matrix, np.full_like(matrix, UNK_IX)])

    def as_matrix(self, sequences, max_len):
        """ Convert a list of tokens into a matrix with padding """
        if isinstance(sequences[0], str):
            sequences = list(map(str.split, sequences))
        lengths = []

        matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))
        for i, seq in enumerate(sequences):
            row_ix = [self.token_to_id.get(word, UNK_IX) for word in seq[:max_len]]
            matrix[i, :len(row_ix)] = row_ix
            lengths.append(len(row_ix))
        return matrix, lengths

    def to_tensors(self, batch):
        batch_tensors = dict()
        for key, arr in batch.items():
            if key in TEXT_COLUMNS:
                batch_tensors[key] = torch.tensor(arr, device=self.device, dtype=torch.int64)
            else:
                batch_tensors[key] = torch.tensor(arr, device=self.device)
        return batch_tensors

    def make_batch(self, batch_data, word_dropout=0):
        """
        Creates a keras-friendly dict from the batch data.
        :param batch_data: raw data to create batch from
        :param word_dropout: replaces token index with UNK_IX with this probability
        :returns: a dict with {'title' : int64[batch, title_max_len]
        """
        batch = {}

        batch["Title"], batch['Title_len'] = self.as_matrix(batch_data["Title"].values, self.title_max_len)
        batch["FullDescription"], batch['Desc_len'] = self.as_matrix(batch_data["FullDescription"].values,
                                                                     self.desc_max_len)
        batch['Categorical'] = self.categorical_vectorizer.transform(
            batch_data[CATEGORICAL_COLUMNS].apply(dict, axis=1))

        if word_dropout != 0:
            batch["FullDescription"] = self.apply_word_dropout(batch["FullDescription"], 1. - word_dropout)

        if TARGET_COLUMN in batch_data.columns:
            batch[TARGET_COLUMN] = batch_data[TARGET_COLUMN].values

        return self.to_tensors(batch)

    def iterate_minibatches(self, data, batch_size=256, shuffle=True, cycle=False, **kwargs):
        """ iterates minibatches of data in random order """
        while True:
            indices = np.arange(len(data))
            if shuffle:
                indices = np.random.permutation(indices)

            for start in range(0, len(indices), batch_size):
                batch = self.make_batch(data.iloc[indices[start: start + batch_size]], **kwargs)
                yield batch

            if not cycle: break
