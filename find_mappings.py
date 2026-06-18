import argparse
import pandas as pd
from SelectLabelWords.Dataset import Dataset
from MappingSelector.MappingSelector import MappingSelector


parser = argparse.ArgumentParser("")
parser.add_argument("--task", type=str, default="parsinlu-food-sentiment")
parser.add_argument("--model_name_or_path", type=str, default="xlm-roberta-large")
parser.add_argument("--template", type=str)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--label_words", type=str)
parser.add_argument("--del_a_last_char", action="store_true")
parser.add_argument("--del_b_last_char", action="store_true")
parser.add_argument("--file_counter", type=str)
args = parser.parse_args()


if args.task == "digikala-text-classification":
    dataset = Dataset("./data/digikala-text-classification", text_a_column=0, text_b_column=None, label_column=1)
elif args.task == "parsi-nlu-foodsentiment":
    dataset = Dataset("./data/parsi-nlu-foodsentiment", text_a_column=1, text_b_column=None, label_column=0)
elif args.task == "parsi-nlu-moviesentiment":
    dataset = Dataset("./data/parsi-nlu-moviesentiment", text_a_column=1, text_b_column=None, label_column=0)
elif args.task == "parsi-nlu-nli":
    dataset = Dataset("./data/parsi-nlu-nli",  text_a_column=0, text_b_column=1, label_column=2)
else:
    raise NotImplementedError

label_words = eval(args.label_words)
output_path = f"./mappings/{args.task}-{args.file_counter}.txt"

mapping_selector = MappingSelector(
    model_name=args.model_name_or_path,
    dataset=dataset,
    label_words=label_words,
    template=args.template,
    del_a_last_char=args.del_a_last_char,
    del_b_last_char=args.del_b_last_char,
    output_path=output_path,
    max_length=args.max_seq_length
).score_mappings()

