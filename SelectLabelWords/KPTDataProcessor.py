from openprompt.data_utils.data_processor import DataProcessor
import os
import pandas as pd
from openprompt.data_utils.utils import InputExample


class ParsinluSentimentProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["خوب", "بد", "متوسط"]
        self.label_column_to_ids = {"Positive": 0, "Negative": 1, "Neutral": 2}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        dataset = pd.read_csv(path, header=None)
        for index, row in dataset.iterrows():
            label, sentence = row[0], row[1]
            example = InputExample(guid=str(index), text_a=sentence, label=self.label_column_to_ids[row[0]])
            examples.append(example)
        return examples


class ParsiNLUNLI(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["بله", "شاید", "خیر"]
        self.label_column_to_ids = {"e": 0, "c": 2, "n": 1}
        self.punctuations = ["،","؛",":","؟","!",".","—","-","%"]

    def get_train_examples(self, data_dir, del_a_last_char=False, del_b_last_char=False) -> InputExample:
        return self.get_examples(data_dir, "train", del_a_last_char, del_b_last_char)

    def get_test_examples(self, data_dir, del_a_last_char=False, del_b_last_char=False) -> InputExample:
        return self.get_examples(data_dir, "test", del_a_last_char, del_b_last_char)

    def get_examples(self, data_dir, split, del_a_last_char=False, del_b_last_char=False):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        dataset = pd.read_csv(path, header=None)
        for index, row in dataset.iterrows():
            label, sentence_a, sentence_b = row[2], row[0], row[1]
            sentence_a = sentence_a[0:-1] if del_a_last_char else sentence_a
            sentence_b = sentence_b[0:-1] if del_b_last_char else sentence_b
            example = InputExample(guid=str(index), text_a=sentence_a, text_b=sentence_b,
                                   label=self.label_column_to_ids[label])
            examples.append(example)
        return examples


class DigikalaTextClassificationProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["فناوری", "بازی","سینما", "سلامتی","ادبیات", "خرید", "عمومی"]
        self.label_column_to_ids = {"علم و تکنولوژی": 0, "بازی ویدیویی": 1, "هنر و سینما": 2, "سلامت و زیبایی": 3,
                                    "کتاب و ادبیات":4, "راهنمای خرید": 5, "عمومی": 6}
        self.punctuations = ["،","؛",":","؟","!",".","—","-","%"]

    def get_train_examples(self, data_dir, del_a_last_char=False, del_b_last_char=False) -> InputExample:
        return self.get_examples(data_dir, "train", del_a_last_char, del_b_last_char)

    def get_test_examples(self, data_dir, del_a_last_char=False, del_b_last_char=False) -> InputExample:
        return self.get_examples(data_dir, "test", del_a_last_char, del_b_last_char)

    def get_examples(self, data_dir, split, del_a_last_char=False, del_b_last_char=False):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        dataset = pd.read_csv(path, header=None)
        for index, row in dataset.iterrows():
            label, sentence_a = row[1], row[0]
            sentence_a = sentence_a[0:-1] if del_a_last_char else sentence_a
            example = InputExample(guid=str(index), text_a=sentence_a, label=self.label_column_to_ids[label])
            examples.append(example)
        return examples
