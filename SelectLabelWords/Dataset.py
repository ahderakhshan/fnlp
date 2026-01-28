import pandas as pd
import os


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
        dataset = pd.read_csv(self.path, header=None)
        result = []
        for index, row in dataset.iterrows():
            if index > 100:
                break
            text_a = row[self.text_a_column]
            text_b = None if self.text_b_column is None else row[self.text_b_column]
            label = row[self.label_column]
            result.append(Data(text_a, label, text_b))
        return result


class Dataset:
    def __init__(self, path):
        self.train = DataSplit(path, "train")
        self.test = DataSplit(path, "test")
        self.dev = DataSplit(path, "dev")
