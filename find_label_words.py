import logging
from SelectLabelWords.SelectBestSamples import ScoreSamples
from SelectLabelWords.Dataset import Dataset
from SelectLabelWords.LabelWordsExplorer import LabelWordsExplorer
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, KnowledgeableVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.plms import ModelClass, _MODEL_CLASSES, MLMTokenizerWrapper
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForMaskedLM
from SelectLabelWords.KPTDataProcessor import ParsinluSentimentProcessor
import argparse
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
args = parser.parse_args()

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}
if args.task == "parsinlu-food-sentiment":
    TemplatePath = "./templates/parsi-nlu-foodsentiment/templates.txt"
    KPTTemplatePath = "./templates/parsi-nlu-foodsentiment/templates_kpt.txt"
    LabelWordPath = "./labelwords/parsi-nlu-foodsentiment/labelwords.txt"
    LanguageModel = args.model_name_or_path
    OutputPath = "./templates/parsi-nlu-foodsentiment/explored_templates.txt"
    main_label_words = {"Positive": "خوب", "Negative": "بد", "Neutral": "متوسط"}
    dataset['train'] = ParsinluSentimentProcessor().get_train_examples("./data/parsi-nlu-foodsentiment/")
    dataset['test'] = ParsinluSentimentProcessor().get_test_examples("./data/parsi-nlu-foodsentiment/")
    class_labels = ParsinluSentimentProcessor().get_labels()
    batch_s = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

my_dataset = Dataset("./data/parsi-nlu-foodsentiment", text_a_column=1, text_b_column=None, label_column=0)
sample_selector = ScoreSamples(template_path=TemplatePath,
                               label_words_path=LabelWordPath,
                               language_model_path=LanguageModel,
                               dataset=my_dataset,
                               write_sample_scores=True,
                               output_path="./data/parsi-nlu-foodsentiment/train_scores.csv",
                               max_length=128)
sample_selector.score_samples()
my_dataset = sample_selector.dataset
with open(TemplatePath, 'r') as f:
    templates = f.read().strip().split("\n")

explored_label_words = []
for template_index, template in enumerate(templates):
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
    logging.info(f"label words found for template: {template} are {finded_label_words}")
    if len(args.filters) > 0:
        mytemplate = ManualTemplate(tokenizer=tokenizer, multi_token_handler="mean").from_file(KPTTemplatePath,
                                                                                               choice=template_index)
        myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=args.cutoff,
                                               max_token_split=args.max_seq_length)
        myverbalizer.label_words = finded_label_words.values()
        support_dataset = dataset['train']
        for example in support_dataset:
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
        if "RR" in args.filters:
            record = tfidf_filter(myverbalizer, cc_logits, class_labels)
            logging.info(f"after RR number of label words per classes {[len(i) for i in myverbalizer.label_words]}")
            logging.info(f"after RR label words are {myverbalizer.label_words}")

    explored_label_words.append(myverbalizer.label_words)

with open(OutputPath, "w") as f:
    for explored_label_word in explored_label_words:
        f.write(str(explored_label_word) + "\n")

logging.info("Complete")





