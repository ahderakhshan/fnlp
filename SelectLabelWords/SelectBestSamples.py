import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F
import logging


class ScoreSamples:
    def __init__(self, template_path, label_words_path, language_model_path, dataset, threshold=0.5, max_length=512,
                 write_sample_scores=False, output_path=None):
        self.templates = self.load_templates(template_path)
        self.label_words = self.load_label_words(label_words_path)
        self.language_model = AutoModelForMaskedLM.from_pretrained(language_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("language model loaded successfully")
        self.language_model.eval()
        self.language_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_path)
        logging.info("tokenizer loaded successfully")
        self.dataset = dataset
        self.threshold = threshold
        self.max_length = max_length
        self.write_sample_scores = write_sample_scores
        self.output_path = output_path

    @staticmethod
    def load_templates(template_path):
        with open(template_path, 'r') as f:
            lines = f.read().strip().split("\n")
        return lines

    @staticmethod
    def load_label_words(label_words_path):
        with open(label_words_path, 'r') as f:
            lines = f.read().strip().split("\n")
            return [eval(line) for line in lines]

    def convert_label_word_to_ids(self, label_word):
        result = {}
        for key, value in label_word.items():
            value_id = self.tokenizer.encode(value, add_special_tokens=False)
            if len(value_id) == 1:
                result[key] = value_id
            else:
                raise ValueError(f"initial label words must be single token {value} for {key} is unacceptable")
        return result

    def score_samples(self):
        logging.info("start computing scores")
        for label_word in self.label_word:
            label_word_ids = self.convert_label_word_to_ids(label_word)
            for template in self.templates:
                for sample in self.dataset.train.data:
                    predicted_label = self.predict(sample, template, label_word_ids)
                    if predicted_label == sample.label:
                        sample.score += 1
                logging.info(f"socres for template {template} and label words {label_word} computed")
        statements = len(self.templates) * len(self.label_words)
        for sample in self.dataset.train.data:
            sample.score /= statements
        if self.write_sample_scores:
            self.write_samples()

    def write_samples(self):
        dataframe = {"text_a": [], "label": [], "scores": []}
        if self.dataset.train.data[0].text_b is not None:
            dataframe["text_b"] = []
        for sample in self.dataset.train.data:
            dataframe["text_a"].append(sample.text_a)
            dataframe["label"].append(sample.label)
            dataframe["scores"].append(sample.score)
            if sample.text_b is not None:
                dataframe["text_b"].append(sample.text_b)
        dataframe = pd.DataFrame(dataframe)
        dataframe.to_csv(self.output_path)

    def predict(self, sample, template, label_word_ids):
        template = template.replace("<text_a>", sample.text_a)
        if '<text_b>' in template:
            template = template.replace("<text_b>", sample.text_b)

        tokenized_input = self.tokenizer(template, return_tensors="pt", padding="max_length", max_length=self.max_length
                                         , truncation=True)
        tokenized_input = {k: v.to(self.device) for k, v in tokenized_input.items()}
        input_ids = tokenized_input["input_ids"]
        mask_token_id = self.tokenizer.mask_token_id
        mask_index = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]

        with torch.no_grad():
            outputs = self.language_model(**tokenized_input)
            logits = outputs.logits
        mask_logits = logits[0, mask_index, :]
        probs = F.softmax(mask_logits, dim=-1)

        max_prob = 0
        selected_label = None

        for key, value in label_word_ids.items():  # key is label in dataset and value is label word
            prob = probs[0, value[0]].item()
            if prob >= max_prob:
                max_prob = prob
                selected_label = key

        return selected_label
