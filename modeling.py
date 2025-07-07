import sys
import os
import torch
from torch import nn
import numpy as np
from transformers import BertModel, AutoModel, AutoConfig, AutoTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from tqdm import tqdm


    
import os
import torch
import logging
import numpy as np
from torch import nn
from numpy import ndarray
from torch import nn, Tensor
from tqdm.autonotebook import trange
from typing import List, Dict, Union, Tuple

from transformers import AutoTokenizer, BertPreTrainedModel, BertModel, T5EncoderModel
logger = logging.getLogger(__name__)

    
def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch



class DREncoder(nn.Module):
    def __init__(self, config, init_model=None, vector_dim=768, tokenizer=None):
        super().__init__()
        
        print("#> Building DREncoder...")
        self.encoder = AutoModel.from_pretrained(init_model, config=config)
        print("#> Loading model from {}".format(init_model))
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(init_model)
            
        self.output_fc = nn.Linear(self.encoder.config.hidden_size, vector_dim)
        self.config = config
        self.vector_dim = vector_dim
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.max_input_length = 180

        print("-" * 40)
        print("#> DREncoder:")
        print("#>  - vector_dim: {}".format(self.vector_dim))
        print("#>  - num_non_embedding_params: {} M".format(sum(p.numel() for p in self.non_embedding_parameters() if p.requires_grad) / 1024 / 1024))
        print("#>  - num_total_params: {} M".format(sum(p.numel() for p in self.parameters() if p.requires_grad) / 1024 / 1024))
        print("-" * 40)


    def non_embedding_parameters(self):
        return [p for n, p in self.named_parameters() if 'embedding' not in n]

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_enable(*args, **kwargs)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state

        masked_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.)
        mean_pooling = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        
        text_embeds = self.output_fc(mean_pooling)
        return text_embeds
    
    def batch_encode(self, input_ids, attention_mask, batch_num):
        # batch encoding can reduce the peak CUDA memory when training with grad ckpting
        if batch_num == 1 or input_ids.size(1) < 64:
            return self.encode(input_ids, attention_mask)
        outputs = []
        end = 0
        for i in range(1, batch_num+1):
            start, end = end, int(i/batch_num*len(input_ids))
            out = self.encode(
                input_ids[start:end], attention_mask[start:end]
            )
            outputs.append(out)
        cat_outputs = torch.vstack(outputs)
        assert len(cat_outputs) == len(input_ids)
        return cat_outputs
    
    def forward(self, query, pos, neg):
        query_embeds = self.batch_encode(
            input_ids=query['input_ids'], 
            attention_mask=query['attention_mask'],
            batch_num=8
        )

        pos_embeds = self.batch_encode(
            input_ids=pos['input_ids'],
            attention_mask=pos['attention_mask'],
            batch_num=8
        )

        neg_embeds = self.batch_encode(
            input_ids=neg['input_ids'],
            attention_mask=neg['attention_mask'],
            batch_num=8
        )
        
        positive_scores = (query_embeds * pos_embeds).sum(dim=1, keepdim=True)
        negative_scores = (query_embeds.unsqueeze(1) * neg_embeds.unsqueeze(0)).sum(dim=-1)
        scores = torch.cat([positive_scores, negative_scores], dim=-1)

        labels = torch.zeros(len(scores), dtype=torch.long, device=scores.device)
        loss = self.loss_fn(scores, labels)

        return {"loss": loss, "scores": scores}


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("./init_models/bert_uncased_L-2_H-128_A-2")
    config = AutoConfig.from_pretrained("./init_models/bert_uncased_L-2_H-128_A-2")
    model = DREncoder(
        config=config,
        init_model="./init_models/bert_uncased_L-2_H-128_A-2",
        vector_dim=768,
        tokenizer=tokenizer
    )
    
    text1 = "What is the capital of California?"
    text2 = "The capital of California is Sacramento."

    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True)
    outputs = model.encode(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    print(outputs[:, :5])
