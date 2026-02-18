import pandas as pd
import os
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, text_a, label, text_b=None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.score = 0  # this score is used to select best samples


class DataSplit:
    def __init__(self, path, split, text_a_column=0, text_b_column=None, label_column=1):
        if split not in ["train", "test", "dev"]:
            raise ValueError("split must be train, test and dev")
        self.path = os.path.join(path, split + ".csv")
        self.text_a_column = text_a_column
        self.text_b_column = text_b_column
        self.label_column = label_column
        self.data = self.load_data()

    def load_data(self):
        dataset = pd.read_csv(self.path, header=None, on_bad_lines="skip")
        sample_size = min(2000, dataset.shape[0])  # Total desired sample size
        sampled_dataset, _ = train_test_split(
            dataset,
            train_size=0.3,
            stratify=dataset[self.label_column],
            random_state=42
        )
        result = []
        for index, row in dataset.iterrows():
            text_a = row[self.text_a_column]
            text_b = None if self.text_b_column is None else row[self.text_b_column]
            label = row[self.label_column]
            result.append(Data(text_a, label, text_b))
        return result


class Dataset:
    def __init__(self, path, text_a_column, text_b_column, label_column):
        self.train = DataSplit(path, "train", text_a_column=text_a_column, text_b_column=text_b_column,
                               label_column=label_column)
        self.test = DataSplit(path, "test", text_a_column=text_a_column, text_b_column=text_b_column,
                              label_column=label_column)
        self.dev = DataSplit(path, "dev", text_a_column=text_a_column, text_b_column=text_b_column,
                             label_column=label_column)
