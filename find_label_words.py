import logging
from SelectLabelWords.SelectBestSamples import ScoreSamples
from SelectLabelWords.Dataset import Dataset
from SelectLabelWords.LabelWordsExplorer import LabelWordsExplorer

TemplatePath = "./templates/parsi-nlu-foodsentiment/templates.txt"
LabelWordPath = "./labelwords/parsi-nlu-foodsentiment/labelwords.txt"
LanguageModel = "/home/am_derakhshan/fnlp/models/xlm-roberta/xlm-roberta-large/"
OutputPath = "./templates/parsi-nlu-foodsentiment/explored_templates.txt"

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

main_label_words = {"Positive": "خوب", "Negative": "بد", "Neutral": "متوسط"}
explored_label_words = []
for template in templates:
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
                                             max_length=256)
    explored_label_words.append(label_word_explorer.find_label_words_single_token())

with open(OutputPath, "w") as f:
    for explored_label_word in explored_label_words:
        f.write(str(explored_label_word) + "\n")

logging.info("Complete")





