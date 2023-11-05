import numpy as np

from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from typing import List

def measure_toxicity(texts: List[str], batch_size: int = 32):
    res_labels = []
    res_scores = []

    tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = tokenizer(texts[i:i + batch_size], return_tensors='pt', padding=True)

        labels = model(**batch)['logits'].argmax(1).float().data.tolist()
        res_labels.extend(labels)

        logits_tensors = model(**batch)['logits'].float().data
        logits_tensors = logits_tensors[:, 0] - logits_tensors[:, 1]
        logits_list = logits_tensors.view(-1, 1).tolist()
        res_scores.extend(logits_list)

    return res_labels, res_scores