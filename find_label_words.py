import logging
from SelectLabelWords.SelectBestSamples import ScoreSamples
from SelectLabelWords.Dataset import Dataset
from SelectLabelWords.LabelWordsExplorer import LabelWordsExplorer
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, KnowledgeableVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.plms import ModelClass, _MODEL_CLASSES, MLMTokenizerWrapper
from openprompt.data_utils.data_sampler import FewShotSampler
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForMaskedLM
from SelectLabelWords.KPTDataProcessor import ParsinluSentimentProcessor, ParsiNLUNLI,\
    DigikalaTextClassificationProcessor
import argparse
import torch
from SelectLabelWords.utils import calibrate, tfidf_filter
_MODEL_CLASSES['xlmroberta'] = ModelClass(**{
    'config': XLMRobertaConfig,
    'tokenizer': XLMRobertaTokenizer,
    'model': XLMRobertaForMaskedLM,
    'wrapper': MLMTokenizerWrapper,
})

parser = argparse.ArgumentParser("")
parser.add_argument("--task", type=str, default="parsinlu-food-sentiment")
parser.add_argument("--filters", type=str, nargs="*", default=["FR", "RR"])
parser.add_argument("--model_name_or_path", type=str,
                    default="/home/am_derakhshan/fnlp/models/xlm-roberta/xlm-roberta-large/")
parser.add_argument("--model", type=str, default="xlmroberta")
parser.add_argument("--cutoff", type=float, default=0.8)
parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}
if args.task == "parsinlu-food-sentiment":
    DataPath = "./data/parsi-nlu-foodsentiment/"
    ScoreOutputPath = "./data/parsi-nlu-foodsentiment/train_scores.csv"
    TemplatePath = "./templates/parsi-nlu-foodsentiment/templates.txt"
    KPTTemplatePath = "./templates/parsi-nlu-foodsentiment/templates_kpt.txt"
    LabelWordPath = "./labelwords/parsi-nlu-foodsentiment/labelwords.txt"
    LanguageModel = args.model_name_or_path
    OutputPath = "./labelwords/parsi-nlu-foodsentiment/explored_labelwrods_fodd.txt"
    main_label_words = {"Positive": "خوب", "Negative": "بد", "Neutral": "متوسط"}
    dataset['train'] = ParsinluSentimentProcessor().get_train_examples(DataPath)
    dataset['test'] = ParsinluSentimentProcessor().get_test_examples(DataPath)
    class_labels = ParsinluSentimentProcessor().get_labels()
    del_a_chars = None
    del_b_chars = None
    batch_s = 1

elif args.task == "parsinlu-movie-sentiment":
    DataPath = "./data/parsi-nlu-moviesentiment/"
    ScoreOutputPath = "./data/parsi-nlu-moviesentiment/train_scores.csv"
    TemplatePath = "./templates/parsi-nlu-foodsentiment/templates.txt"
    KPTTemplatePath = "./templates/parsi-nlu-foodsentiment/templates_kpt.txt"
    LabelWordPath = "./labelwords/parsi-nlu-foodsentiment/labelwords.txt"
    LanguageModel = args.model_name_or_path
    OutputPath = "./labelwords/parsi-nlu-foodsentiment/explored_labelwords_movie.txt"
    main_label_words = {"Positive": "خوب", "Negative": "بد", "Neutral": "متوسط"}
    dataset['train'] = ParsinluSentimentProcessor().get_train_examples(DataPath)
    dataset['test'] = ParsinluSentimentProcessor().get_test_examples(DataPath)
    class_labels = ParsinluSentimentProcessor().get_labels()
    del_a_chars = None
    del_b_chars = None
    batch_s = 1

elif args.task == "parsinlu-nli":
    DataPath = "./data/parsi-nlu-nli/"
    ScoreOutputPath = "./data/parsi-nlu-nli/train_scores.csv"
    TemplatePath = "./templates/parsi-nlu-nli/templates.txt"
    KPTTemplatePath = "./templates/parsi-nlu-nli/templates_kpt.txt"
    LabelWordPath = "./labelwords/parsi-nlu-nli/labelwords.txt"
    LanguageModel = args.model_name_or_path
    OutputPath = "./labelwords/parsi-nlu-nli/explored_labelwords.txt"
    main_label_words = {"e": "بله", "n": "شاید", "c": "خیر"}
    del_a_chars = [True, False, True]
    del_b_chars = [False, False, True]
    batch_s = 1

elif args.task == "digikala-tc":
    DataPath = "./data/digikala-text-classification/"
    ScoreOutputPath = "./data/digikalal-text-classification/train_scores.csv"
    TemplatePath = "./templates/digikalal-text-classification/templates.txt"
    KPTTemplatePath = "./templates/digikalal-text-classification/templates_kpt.txt"
    LabelWordPath = "./labelwords/digikalal-text-classification/labelwords.txt"
    LanguageModel = args.model_name_or_path
    OutputPath = "./labelwords/digikalal-text-classification/explored_labelwords.txt"
    #main_label_words = {"Positive": "خوب", "Negative": "بد", "Neutral": "متوسط"}
    del_a_chars = [False, True]
    del_b_chars = [False, False]
    batch_s = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

my_dataset = Dataset(DataPath, text_a_column=1, text_b_column=None, label_column=0)
sample_selector = ScoreSamples(template_path=TemplatePath,
                               label_words_path=LabelWordPath,
                               language_model_path=LanguageModel,
                               dataset=my_dataset,
                               write_sample_scores=True,
                               output_path=ScoreOutputPath,
                               max_length=128,
                               del_a_chars=del_a_chars,
                               del_b_chars=del_b_chars)
sample_selector.score_samples()
my_dataset = sample_selector.dataset
with open(TemplatePath, 'r') as f:
    templates = f.read().strip().split("\n")

output_file = open(OutputPath, "w")
explored_label_words = []
for template_index, template in enumerate(templates):
    f.write(f"initial label words for template {template} \n")
    label_word_explorer = LabelWordsExplorer(language_model=LanguageModel,
                                             dataset=my_dataset,
                                             initial_label_words=main_label_words,
                                             template=template,
                                             threshold=0.5,
                                             k1=50,
                                             m1=20,
                                             k2=25,
                                             m2=20,
                                             n2=30,
                                             max_length=512)
    finded_label_words = label_word_explorer.find_label_words()
    for key, values in finded_label_words.items():
        f.write(f"\t label words for {key} \n")
        for val in values:
            f.write(f"\t\t {val} \n")
    logging.info(f"label words found for template: {template} are {finded_label_words}")
    if len(args.filters) > 0:
        if args.task == "parsinlu-nli":
            if template_index == 0:
                dataset['train'] = ParsiNLUNLI().get_train_examples(DataPath, del_a_last_char=True)
                dataset['test'] = ParsiNLUNLI().get_test_examples(DataPath, del_a_last_char=True)
                class_labels = ParsiNLUNLI().get_labels()
            elif template_index == 1:
                dataset['train'] = ParsiNLUNLI().get_train_examples(DataPath)
                dataset['test'] = ParsiNLUNLI().get_test_examples(DataPath)
                class_labels = ParsiNLUNLI().get_labels()
            elif template_index == 2:
                dataset['train'] = ParsiNLUNLI().get_train_examples(DataPath, del_a_last_char=True, del_b_last_char=True)
                dataset['test'] = ParsiNLUNLI().get_test_examples(DataPath, del_a_last_char=True, del_b_last_char=True)
                class_labels = ParsiNLUNLI().get_labels()
        if args.task == "digikala-tc":
            if template_index == 0:
                dataset['train'] = DigikalaTextClassificationProcessor().get_train_examples(DataPath)
                dataset['test'] = DigikalaTextClassificationProcessor().get_test_examples(DataPath)
                class_labels = DigikalaTextClassificationProcessor().get_labels()
            elif template_index == 1:
                dataset['train'] = DigikalaTextClassificationProcessor().get_train_examples(DataPath, del_a_last_char=True)
                dataset['test'] = DigikalaTextClassificationProcessor().get_test_examples(DataPath, del_a_last_char=True)
                class_labels = DigikalaTextClassificationProcessor().get_labels()

        mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(KPTTemplatePath, choice=template_index)
        myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=args.cutoff,
                                               multi_token_handler="mean", max_token_split=args.max_seq_length)
        myverbalizer.label_words = list(finded_label_words.values())
        # support_sampler = FewShotSampler(num_examples_per_label=70, also_sample_dev=False)
        dataset['support'] = dataset["train"]  # support_sampler(dataset['train'], seed=args.seed)
        support_dataset = dataset['support']
        for example in support_dataset:
            # print(f"example.label is {example.label}")
            example.label = -1
        support_dataloader = PromptDataLoader(dataset=support_dataset, template=mytemplate, tokenizer=tokenizer,
                                              tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length,
                                              decoder_max_length=3, batch_size=batch_s, shuffle=False,
                                              teacher_forcing=False, predict_eos_token=False, truncate_method="tail")

        prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                               plm_eval_mode=args.plm_eval_mode)
        prompt_model = prompt_model.cuda()
        org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
        cc_logits = calibrate(prompt_model, support_dataloader)
        if "FR" in args.filters:
            myverbalizer.register_calibrate_logits(cc_logits.mean(dim=0))
            logging.info(f"after FR number of label words per classes {[len(i) for i in myverbalizer.label_words]}")
            logging.info(f"after FR label words are {myverbalizer.label_words}")
        f.write(f"after Frequency Refinement label words for template {template} \n")
        for label_num, label_words in myverbalizer.label_words:
            f.write(f"\t for label {list(main_label_words.keys())[label_num]}\n")
            for word in label_words:
                f.write(f"\t\t {word}\n")
        # for i in range(len(myverbalizer.label_words)):
        #     myverbalizer.label_words[i] = [list(main_label_words.values())[i]] + myverbalizer.label_words[i]
        logging.info(f"before RR label words are {myverbalizer.label_words}")
        if "RR" in args.filters:
            record = tfidf_filter(myverbalizer, cc_logits, class_labels)
            logging.info(f"after RR number of label words per classes {[len(i) for i in myverbalizer.label_words]}")
            logging.info(f"after RR label words are {myverbalizer.label_words}")

        f.write(f"after Relevance Refinement label words for template {template} \n")
        for label_num, label_words in myverbalizer.label_words:
            f.write(f"\t for label {list(main_label_words.keys())[label_num]} \n")
            for word in label_words:
                f.write(f"\t\t {word} \n")

        f.write(f"********************************************\n")

    explored_label_words.append(myverbalizer.label_words)
    del label_word_explorer
    torch.cuda.empty_cache()

# with open(OutputPath, "w") as f:
#     for explored_label_word in explored_label_words:
#         f.write(str(explored_label_word) + "\n")
f.close()
logging.info("Complete")





