import numpy as np
import pandas as pd
from collections import Counter
from typing import List
import matplotlib.pyplot as plt


def get_words_from_text_column(data: pd.DataFrame, column: str):
    vals = data[column].apply(lambda x: x.split(' ')).values
    return [item for sublist in list(vals) for item in sublist]


def get_frequent_tokens(data: pd.DataFrame, text_columns: List[str], min_count: int):
    words = []
    for col in text_columns:
        words.extend(get_words_from_text_column(data, col))
    token_counts = Counter(words)
    tokens = sorted(t for t, c in token_counts.items() if c >= min_count)
    return tokens


def get_embedding_matrix(model, tokens):
    words = list(model.vocab.keys())
    word_vectors = np.array([model[w] for w in words])
    embedding_matrix = np.zeros(shape=(len(tokens), 100)).astype(np.float32)
    for i in range(len(tokens)):
        word = tokens[i]
        if word in model:
            embedding_matrix[i] = model[word]
        else:
            if word == 'UNK':
                embedding_matrix[i] = (np.zeros(100, )).astype(np.float32)
            elif word == 'PAD':
                embedding_matrix[i] = np.mean(word_vectors, axis=0)
            else:
                embedding_matrix[i] = (np.random.rand(100, ) / np.sqrt(len(tokens))).astype(np.float32)
    return embedding_matrix


def get_metric_epoch(metrics_data, metric_name):
    return [x[metric_name] for x in metrics_data]


def plot_one_metric(metrics_data, ax, metric_name):
    n_epoch = len(metrics_data)
    epochs = range(n_epoch)

    val_metric = get_metric_epoch(metrics_data, f'val_{metric_name}')
    train_metric = get_metric_epoch(metrics_data, f'train_{metric_name}')

    ax.plot(epochs, train_metric, label=f'Training {metric_name}')
    ax.plot(epochs, val_metric, label=f'Validation {metric_name}')

    ax.set_title(f'Training and Validation {metric_name}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_name)

    ax.set_xticks(np.arange(0, n_epoch, 1))
    ax.legend(loc='best')


def plot_metric(metric_data, network_description):
    fig, axs = plt.subplots(3, figsize=(10, 20))
    fig.suptitle(f'Plotting results for {network_description}')
    plot_one_metric(metric_data, axs[0], 'loss')
    plot_one_metric(metric_data, axs[1], 'mse')
    plot_one_metric(metric_data, axs[2], 'mae')
