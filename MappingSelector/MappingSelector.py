from itertools import product
from SelectLabelWords.Dataset import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import logging
import os


class MappingSelector:
    def __init__(self, model_name, dataset: Dataset, label_words: dict, template: str, del_a_last_char, del_b_last_char,
                 output_path, max_length=512, batch_size=128):
        self.dataset = dataset
        self.label_words = label_words
        self.template = template
        self.del_a_last_char = del_a_last_char
        self.del_b_last_char = del_b_last_char
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_path = output_path
        self.model.to(self.device)
        # just 1 token label words must remain in this phase
        self.label_words = {k: [word for word in v if len(self.tokenizer.tokenize(word)) == 1]
                            for k, v in self.label_words.items()}
        print(f"label words for just 1 tokens {self.label_words}")
        self.puncs = ["،", ".", "؟", "!", ":"]
        self.batch_size = batch_size

    def score_mappings(self):
        print(f"start score mapping")
        res = 1
        for value in self.label_words.values():
            res*=len(value)
        print(f"len all mappings is {res}")
        all_mappings = [
            dict(zip(self.label_words.keys(), values))
            for values in product(*self.label_words.values())
        ]
        correct_labels = [data.label for data in self.dataset.train.data]
        all_predictions = self.get_predictions(all_mappings)
        all_acc = []
        print(f"len all mappings is {len(all_mappings)}")
        for index, prediction in enumerate(all_predictions):
            print(f"start computing {all_mappings[index]} score")
            mapping_accuracy = accuracy_score(correct_labels, prediction)
            all_acc.append(mapping_accuracy)
            print(f"accuracy for {all_mappings[index]} = {all_acc[-1]}")
        argsorts = sorted(range(len(all_acc)), key=lambda i: all_acc[i], reverse=True)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for index in argsorts:
                f.write(f"{all_mappings[index]}---{all_acc[index]}\n")

    def get_predictions(self, all_mappings):
        all_predictions = [[] for i in range(len(all_mappings))]
        batch_size = self.batch_size
        batched_data = [self.dataset.train.data[i:i+batch_size] for i in range(0,len(self.dataset.train.data),batch_size)]
        for data in batched_data:
            all_templates = [self.template.replace("<text_a>", sample.text_a[:-1]
                if self.del_a_last_char and sample.text_a[-1] in self.puncs else sample.text_a) for sample in data]
            #template = self.template.replace("<text_a>", data.text_a[:-1] if self.del_a_last_char and data.text_a[-1] in self.puncs else data.text_a)
            if data[0].text_b:
                all_templates = [template.replace("<text_b>", sample.text_b[:-1] if self.del_b_last_char and sample.text_b[-1] in self.puncs else sample.text_b)
                                 for template, sample in zip(all_templates, data)]
            template_tokens = self.tokenizer(all_templates, return_tensors="pt", padding="max_length",
                                                      max_length=self.max_length, truncation=True)
            tokenized_input = {k: v.to(self.device) for k, v in template_tokens.items()}

            with torch.no_grad():
                outputs = self.model(**tokenized_input)
                logits = outputs.logits

            input_ids = tokenized_input["input_ids"]
            mask_index = (
                    input_ids == self.tokenizer.mask_token_id
            ).nonzero(as_tuple=True)[1]
            batch_ids = torch.arange(
                logits.shape[0],
                device=self.device
            )
            mask_logits = logits[batch_ids, mask_index, :]
            probs = F.softmax(mask_logits, dim=-1)
            for index, mapping in enumerate(all_mappings):
                label_word_ids = {label: self.tokenizer.encode(word, add_special_tokens=False)
                                  for label, word in mapping.items()}

                for i in range(probs.shape[0]):
                    scores = {
                        label: probs[i, ids[0]].item()
                        for label, ids in label_word_ids.items()
                    }

                    all_predictions[index].append(
                        max(scores, key=scores.get)
                    )

        return all_predictions

