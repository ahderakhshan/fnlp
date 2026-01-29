from SelectLabelWords.SelectBestSamples import ScoreSamples
from SelectLabelWords.Dataset import Dataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


my_dataset = Dataset("./data/parsi-nlu-foodsentiment", text_a_column=1, text_b_column=None, label_column=0)
sample_selector = ScoreSamples(template_path="./templates/parsi-nlu-foodsentiment/templates.txt",
                               label_words_path="./labelwords/parsi-nlu-foodsentiment/labelwords.txt",
                               language_model_path="/home/am_derakhshan/fnlp/models/xlm-roberta/xlm-roberta-large/",
                               dataset=my_dataset,
                               write_sample_scores=True,
                               output_path="./data/parsi-nlu-foodsentiment/train_scores.csv",
                               max_length=512)
sample_selector.score_samples()