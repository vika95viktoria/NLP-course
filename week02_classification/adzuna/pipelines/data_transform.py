import pandas as pd
import numpy as np
import nltk
from typing import List
from collections import Counter


class DataProcessor:
    def __init__(self, csv_location, text_columns: List[str], category_columns: List[str],
                 company_col: str, target_column: str):
        self.data = pd.read_csv(csv_location, compression='zip', index_col=None)
        self.text_columns = text_columns
        self.category_columns = category_columns
        self.target_column = target_column
        self.tokenizer = nltk.tokenize.WordPunctTokenizer()
        self.col_company_name = company_col

    def tokenize_text(self):
        for col in self.text_columns:
            self.data[col] = self.data[col].apply(lambda x: ' '.join(self.tokenizer.tokenize((str(x)).lower())))

    def log_transform(self):
        self.data[self.target_column] = np.log1p(self.data['SalaryNormalized']).astype('float32')

    def process_category_columns(self):
        self.data[self.category_columns] = self.data[self.category_columns].fillna('NaN')

    def process_company_column(self):
        top_companies, top_counts = zip(*Counter(self.data[self.col_company_name]).most_common(1000))
        recognized_companies = set(top_companies)
        self.data[self.col_company_name] = self.data[self.col_company_name].apply(
            lambda comp: comp if comp in recognized_companies else "Other")

    def process_text_columns(self):
        self.tokenize_text()

    def process(self):
        self.log_transform()
        self.process_category_columns()
        self.tokenize_text()
        self.process_company_column()
        return self.data

