import torch
import os

from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List

device = torch.device('cuda:0')


class T5Paraphrase():

    def __init__(self, model_dir_path: str="../models/t5-paraphrase/") -> None:

        self.toxicity_tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.toxicity_model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained("ceshine/t5-paraphrase-paws-msrp-opinosis")
        self.paraphrase_model = self.load_model(model_dir_path)

    def load_model(self, model_dir_path):
        paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("ceshine/t5-paraphrase-paws-msrp-opinosis")

        model_weights_path = os.path.join(model_dir_path, "pytorch_model.bin")
        paraphrase_model.load_state_dict(torch.load(model_weights_path))

        paraphrase_model.to(device)

        return paraphrase_model
    
    def measure_toxicity(self, texts: List[str], batch_size: int = 32):
        res_labels = []
        res_scores = []

        for i in range(0, len(texts), batch_size):
            batch = self.toxicity_tokenizer(texts[i:i + batch_size], return_tensors='pt', padding=True)

            labels = self.toxicity_model(**batch)['logits'].argmax(1).float().data.tolist()
            res_labels.extend(labels)

            logits_tensors = self.toxicity_model(**batch)['logits'].float().data
            logits_tensors = logits_tensors[:, 0] - logits_tensors[:, 1]
            logits_list = logits_tensors.view(-1, 1).tolist()
            res_scores.extend(logits_list)

        return res_labels, res_scores

    def detoxify(
        self,
        sentence: str,
        num_paraphrases: int = 10,
        max_length: int = 64,
        num_beams: int = 10
    ):
        # encode the inputs to the paraphraser model
        encoded_inputs = self.paraphrase_tokenizer(sentence, return_tensors='pt')
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

        # generate paraphrases
        encoded_outputs = self.paraphrase_model.generate(
            **encoded_inputs,
            do_sample=False,
            num_return_sequences=num_paraphrases,
            max_length=max_length,
            num_beams=num_beams
        )
        
        # decode the model outputs
        paraphrases = [self.paraphrase_tokenizer.decode(out, skip_special_tokens=True) for out in encoded_outputs]

        # measure toxicity
        toxic_labels, toxic_scores = self.measure_toxicity(paraphrases)

        # sort the outputs in ascending order of toxicity metric
        sorted_indices = sorted(range(len(toxic_scores)), key=lambda k: toxic_scores[k], reverse=True)
        toxic_labels = [toxic_labels[i] for i in sorted_indices]
        toxic_scores = [toxic_scores[i] for i in sorted_indices]
        paraphrases = [paraphrases[i] for i in sorted_indices]

        best_paraphrase = paraphrases[0]

        return best_paraphrase