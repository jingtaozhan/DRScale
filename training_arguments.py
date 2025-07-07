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
    HfArgumentParser,
    set_seed, 
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from transformers.training_args import default_logdir
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter


@dataclass
class MyTrainingArguments(TrainingArguments):
    encode_batch_num: int=field(default=1)

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    per_device_train_batch_size: int = field(default=128)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=5e-6, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.001, metadata={"help": "Weight decay if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=80.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_first_step: bool = field(default=False)
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=9999999999, metadata={"help": "Save checkpoint every X updates steps."})
    
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=521, metadata={"help": "random seed for initialization"})
    fp16: bool = field(default=False)
    local_rank: int = field(default=-1)
    gradient_checkpointing: bool = field(default=True)
    continue_train: bool = field(default=False)
    local_files_only: bool = field(default="scv2024" in os.getenv("HOME"))
    eval_count: int = field(default=-1)

@dataclass
class DataTrainingArguments:
    max_query_length: int = field(default=25) 
    max_doc_length: int = field(default=180)
    num_negative_docs: int = field(default=256)
    train_instances: int = field(default=50000)
    dataset: str = field(default="msmarco")

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    init_model: str = field(default=None)
    vector_dim: int = field(default=768)
    output_model_name: str = field(default=None)

def specify_savedir_with_params(model_args, data_args, training_args:MyTrainingArguments):

    if model_args.output_model_name is None:
        init_model = model_args.init_model.split("/")[-1]
        dataset = data_args.dataset
        training_instances = data_args.train_instances
        model_name = f"{init_model}_{dataset}_{training_instances}"
    else:
        model_name = model_args.output_model_name

    print("model name: ", model_name)
    time_stamp = time.strftime("%b-%d_%H:%M:%S", time.localtime())
    training_args.output_dir = os.path.join(training_args.output_dir, model_name)
    training_args.logging_dir = os.path.join(training_args.logging_dir, f"{model_name}_{time_stamp}")


from transformers import TrainerCallback
from transformers.integrations import TensorBoardCallback
import torch

def is_main_process(local_rank):
    return local_rank in [-1, 0]


class MyTrainerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
             
    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.config.save_pretrained(args.output_dir)
        model.save_pretrained(args.output_dir)
