import sys
import gc
import os
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5TokenizerFast, Trainer, TrainingArguments
from transformers.file_utils import cached_property
from torch.utils.data import DataLoader
from typing import List, Dict, Union


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0')


class ParaphraseDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x['input_ids'])

    def __getitem__(self, idx):
        assert idx < len(self)
        item = {
            key: val[idx] for key, val in self.x.items()
        }
        item['decoder_attention_mask'] = self.y['attention_mask'][idx]
        item['labels'] = self.y['input_ids'][idx]
        return item


class TrainingArgumentsWrapper(TrainingArguments):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.distributed_state = None

    @cached_property
    def _setup_devices(self):
        return device
    

class DataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
            self,
            features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
        
        batch = self.tokenizer.pad(
            features,
            padding=True,
        )
        y_batch = self.tokenizer.pad(
            {
                'input_ids': batch['labels'],
                'attention_mask': batch['decoder_attention_mask']
            },
            padding=True,
        ) 
        batch['labels'] = y_batch['input_ids']
        batch['decoder_attention_mask'] = y_batch['attention_mask']
        
        return {k: torch.tensor(v) for k, v in batch.items()}


def train(
        data_tsv_path: str='../../data/interim/distinguished.tsv',
        model_dir_path: str='../../models/t5-paraphrase'
    ):
    df = pd.read_csv(data_tsv_path, delimiter='\t')

    tokenizer = T5TokenizerFast.from_pretrained("ceshine/t5-paraphrase-paws-msrp-opinosis")

    df_train, df_test_sets = train_test_split(df, test_size=2000)
    df_test, df_valid = train_test_split(df_test_sets, test_size=1000)
    
    
    x_train = tokenizer(df_train['source'].tolist(), truncation=True)
    y_train = tokenizer(df_train['target'].tolist(), truncation=True)
    x_test = tokenizer(df_test['source'].tolist(), truncation=True)
    y_test = tokenizer(df_test['target'].tolist(), truncation=True)

    train_dataset = ParaphraseDataset(x_train, y_train)
    test_dataset = ParaphraseDataset(x_test, y_test)
    len(train_dataset), len(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=4, drop_last=True, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=4, drop_last=True, shuffle=True, num_workers=8)

    model = T5ForConditionalGeneration.from_pretrained('SkolkovoInstitute/t5-paraphrase-paws-msrp-opinosis-paranmt')

    model.to(device)

    training_args = TrainingArgumentsWrapper(
        output_dir=f'../models/{model_dir_path}',
        overwrite_output_dir=False,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        warmup_steps=300,
        weight_decay=0,
        learning_rate=3e-5,
        logging_dir=f'../models/logs/{model_dir_path}',
        logging_steps=100,
        eval_steps=100,
        evaluation_strategy='steps',
        save_total_limit=1,
        save_steps=5000,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    gc.collect()
    torch.cuda.empty_cache()

    trainer.train()

    trainer.evaluate()

    model.save_pretrained(model_dir_path)







