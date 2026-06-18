from itertools import product
from SelectLabelWords.Dataset import Dataset
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


class MappingSelector:
    def __init__(self, model_name, dataset: Dataset, label_words: dict, template: str, del_a_last_char, del_b_last_char,
                 output_path, max_length=512):
        self.dataset = dataset
        self.label_words = label_words
        self.template = template
        self.del_a_last_char = del_a_last_char
        self.del_b_last_char = del_b_last_char
        self.model = XLMRobertaModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_path = output_path

    def score_mappings(self):
        all_mappings = [
            dict(zip(self.label_words.keys(), values))
            for values in product(*self.label_words.values())
        ]
        correct_labels = [data.label for data in self.dataset.train.data]
        all_acc = []
        for mapping in all_mappings:
            predictions = self.get_predictions(mapping)
            mapping_accuracy = accuracy_score(correct_labels, predictions)
            all_acc.append(mapping_accuracy)
        argsorts = sorted(range(len(all_acc)), key=lambda i: all_acc[i], reverse=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            for index in argsorts:
                f.write(f"{all_mappings[index]}---{all_acc[index]}")

    def get_predictions(self, mapping):
        predictions = []
        for data in self.dataset.train.data:
            template = self.template.replace("<text_a>", data.text_a if not self.del_a_last_char else data.text_a[:-1])
            if data.text_b:
                template = template.replace("<text_b>", data.text_b if not self.del_b_last_char else data.text_b[:-1])
            template_tokens = self.tokenizer.tokenize(template, return_tensors="pt", padding="max_length",
                                                      max_length=self.max_length, truncation=True)
            tokenized_input = {k: v.to(self.device) for k, v in template_tokens.items()}
            input_ids = tokenized_input["input_ids"]
            mask_token_id = self.tokenizer.mask_token_id
            mask_index = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]

            with torch.no_grad():
                outputs = self.model(**tokenized_input)
                logits = outputs.logits
            mask_logits = logits[0, mask_index, :]
            probs = F.softmax(mask_logits, dim=-1)
            label_word_ids = {label: self.tokenizer.encode(word, add_special_tokens=False)
                              for label, word in mapping.items()}
            probs = {label: probs[0, word[0]].item() for label, word in label_word_ids.items()}

            predicted_label = max(probs, key=probs.get)
            predictions.append(predicted_label)
        return predictions

