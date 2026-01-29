import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random
import torch


class LabelWordsExplorer:
    def __init__(self, language_model, dataset, initial_label_words, template, threshold, k1, m1, k2, m2,
                 mask="<mask>", max_length=512):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.language_model = AutoModelForMaskedLM.from_pretrained(language_model)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.language_model.to(self.device)
        self.language_model.eval()
        logging.info("language model loaded successfully")
        self.dataset = dataset
        self.template = template
        self.initial_label_words = initial_label_words
        self.threshold = threshold
        self.k1 = k1
        self.m1 = m1
        self.k2 = k2
        self.m2 = m2
        self.mask = mask
        self.max_length = max_length

    def sample_demonstrations(self):
        result = {}
        for label, label_word in self.initial_label_words.items():
            label_sample = random.sample([sample for sample in self.dataset.train.data if (sample.label == label and
                                                                             sample.score >= self.threshold)], k=1)[0]
            result[label] = label_sample
        return result

    def make_input(self, sample, demonstrations):
        input = ""
        for label, sample in demonstrations.items():
            input += " " + self.template
            input = input.replace("<text_a>", sample.text_a)
            if sample.text_b is not None:
                input = input.replace("<text_b>", sample.text_b)
            input = input.replace(self.mask, self.initial_label_words[label])
        input = self.template + input
        input = input.replace("<text_a>", sample.text_a)
        if sample.text_b is not None:
            input = input.replace("<text_b>", sample.text_b)
        return input

    def get_top_k_tokens(self, model_input):
        inputs = self.tokenizer(model_input, return_tensors="pt", padding="max_length", max_length=self.max_length,
                                truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.language_model(**inputs)
            logits = outputs.logits
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        mask_logits = logits[0, mask_token_index, :]
        probs = torch.softmax(mask_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, self.k1, dim=-1)
        top_tokens = self.tokenizer.convert_ids_to_tokens(top_indices[0])
        return [top_token.replace("Ġ", "") for top_token in top_tokens]

    def find_label_words_single_token(self):
        result = {k: {} for k, _ in self.initial_label_words.items()}
        for sample in self.dataset.train.data:
            demonstrations = self.sample_demonstrations()
            model_input = self.make_input(sample, demonstrations)
            logging.info(f"model input is {model_input}")
            top_k_tokens = self.get_top_k_tokens(model_input)
            for index, top_k_token in enumerate(top_k_tokens):
                try:
                    result[sample.label][top_k_token] += 1 / (index + 1)
                except:
                    result[sample.label][top_k_token] = 1 / (index + 1)
        result = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in result.items()}
        result = {k: v[0:self.m1] for k, v in result.items()}
        return result
