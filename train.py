# coding=utf-8
import sys
sys.path.append("./")
import logging
import os
from tqdm import tqdm
import json
import random
from random import choices
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np
import torch
import random
import pickle
import argparse
import numpy as np 
import torch
import logging
from datetime import datetime
import gzip
import os
import tarfile
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    HfArgumentParser,
    set_seed, 
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BertConfig,
)
from transformers.training_args import default_logdir
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from dataset import MSMARCO_Dataset, get_collate_function, DRDataset
from modeling import DREncoder

from training_arguments import *
import pickle

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    specify_savedir_with_params(model_args, data_args, training_args)

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.init_model,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.init_model,
    )
    
    config.gradient_checkpointing = training_args.gradient_checkpointing
    config.encode_batch_num = training_args.encode_batch_num    

    model = DREncoder(
        config=config,
        init_model=model_args.init_model,
        vector_dim=model_args.vector_dim,
        tokenizer=tokenizer,
    ).cuda()

    print("#> Load train dataset")
    set_seed(training_args.seed)
    if data_args.dataset == "msmarco":
        train_dataset = MSMARCO_Dataset(
            num_instances=data_args.train_instances,
            num_negative_docs=data_args.num_negative_docs,
        )
    
    data_collator = get_collate_function(
        data_args.max_query_length,
        data_args.max_doc_length,
        tokenizer,
        train_dataset
    )

    if training_args.eval_count > 0:
        total_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs // training_args.gradient_accumulation_steps
        training_args.eval_steps = total_steps // training_args.eval_count
        if training_args.eval_steps == total_steps:
            training_args.eval_steps -= training_args.gradient_accumulation_steps
        print("#> Set eval steps to", training_args.eval_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        compute_metrics=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    from evaluate import contrastive_perplexity
    
    min_loss = {
        "human": 100,
    }

    def eval_func(model, min_loss, train_dataset):
        test_loss_dict = {}
        for name in train_dataset.test_pairs:
            test_loss, _ = contrastive_perplexity(model, train_dataset)
            
            print(f"test_loss_{name}", test_loss)
            test_loss_dict[name] = test_loss

        if test_loss_dict["human"] < min_loss["human"]:
            print("Original loss:", min_loss)
            print("New loss:", test_loss_dict)

            for name in test_loss_dict:
                min_loss[name] = test_loss_dict[name]
            
            output_dir = os.path.join(training_args.output_dir, "best")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(output_dir)
            model.config.save_pretrained(output_dir)
            print("Save best model to", output_dir)

            
    trainer.evaluate = lambda ignore_keys: eval_func(model, min_loss, train_dataset)
    trainer.add_callback(MyTrainerCallback())

    trainer.train(None)
    trainer.save_model()  # Saves the tokenizer too for easy upload

if __name__ == "__main__":
    main()
