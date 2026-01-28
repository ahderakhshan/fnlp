from SelectLabelWords.SelectBestSamples import ScoreSamples
from SelectLabelWords.Dataset import Dataset


my_dataset = Dataset("./data/parsi-nlu-foodsentiment")
sample_selector = ScoreSamples(template_path="./templates/parsi-nlu-foodsentiment/templates.txt",
                               label_words_path="./labelwords/parsi-nlu-foodsentiment/labelwords.txt",
                               language_model_path="/home/am_derakhshan/fnlp/models/xlm-roberta/xlm-roberta-large/",
                               dataset=my_dataset,
                               write_sample_scores=True,
                               output_path="./data/parsi-nlu-foodsentiment/train_scores.csv",
                               max_length=128)
sample_selector.score_samples()