import logging
import pathlib, os
import random
import argparse
import pickle
from dataset import MSMARCO_Dataset
from modeling import DREncoder, AutoTokenizer, AutoConfig
import argparse
import numpy as np
from typing import Dict, List, Tuple

from tqdm import tqdm
import json
import torch

from dataset import tokenize
from transformers.trainer_utils import set_seed


def encode_negatives(model: DREncoder, dataset: MSMARCO_Dataset, negative_ids, device: torch.device):
    all_embeddings = torch.zeros(len(negative_ids), 768, dtype=torch.float32, device=device)
    eval_batch_size = 64
    for start_idx in tqdm(range(0, len(negative_ids), 64), desc="Encoding negatives"):
        end_idx = min(start_idx + eval_batch_size, len(negative_ids))
        batch_neg = [dataset.corpus[pid] for pid in negative_ids[start_idx:end_idx]]
        batch_neg = tokenize(batch_neg, model.tokenizer, 180)

        with torch.no_grad():
            embeds = model.batch_encode(
                batch_neg['input_ids'].to(device),
                batch_neg['attention_mask'].to(device),
                8
            )
            all_embeddings[start_idx:end_idx] = embeds
    return all_embeddings


def contrastive_perplexity(model: DREncoder, dataset: MSMARCO_Dataset, negative_pool_size: int, num_repeat: int, device: torch.device):
    num_negatives = dataset.num_negative_docs
    eval_batch_size = 64

    model.eval()

    queries = dataset.test_pairs['human'][0]
    qrels = dataset.test_pairs['human'][1]
    corpus = dataset.corpus
    tokenizer = model.tokenizer

    valid_qids = []
    for qid in queries.keys():
        if len(qrels[qid]) > 0:
            valid_qids.append(qid)

    # valid_qids = valid_qids[:500]

    maybe_positive_doc_ids = set()
    for qid in valid_qids:
        maybe_positive_doc_ids |= set(qrels[qid].keys())

    all_doc_ids = set(corpus.keys())
    negative_doc_ids = list(all_doc_ids - maybe_positive_doc_ids)
    all_negative_embeddings = encode_negatives(
        model, dataset, random.sample(negative_doc_ids, negative_pool_size), device
    )
    assert len(valid_qids) > 0

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    contrast_entropy_list = []

    instances = [(qid, pid) for qid in valid_qids for pid in qrels[qid]]

    # print info
    print(f"Total instances: {len(instances)}")
    print(f"Total valid queries: {len(valid_qids)}")
    print(f"Total negative docs: {len(negative_doc_ids)}")
    print(f"Total positive docs: {len(maybe_positive_doc_ids)}")
    print(f"Num sample_negatives: {num_negatives}")

    for start_idx in tqdm(range(0, len(instances), eval_batch_size), desc="Contrastive perplexity"):
        end_idx = min(start_idx + eval_batch_size, len(instances))
        batch_instances = instances[start_idx:end_idx]
        batch_qids = [x[0] for x in batch_instances]
        batch_pids = [x[1] for x in batch_instances]

        batch_queries = [queries[qid] for qid in batch_qids]
        batch_pos = [corpus[pid] for pid in batch_pids]

        batch_queries = tokenize(batch_queries, tokenizer, 32)
        batch_pos = tokenize(batch_pos, tokenizer, 180)

        with torch.no_grad():
            query_embes = model.batch_encode(
                batch_queries['input_ids'].to(device),
                batch_queries['attention_mask'].to(device),
                8
            )
            pos_embes = model.batch_encode(
                batch_pos['input_ids'].to(device),
                batch_pos['attention_mask'].to(device),
                8
            )
            for i in range(len(batch_qids)):
                query_emb = query_embes[i:i+1]  # (1, 768)
                pos_emb = pos_embes[i:i+1]      # (1, 768)

                positive_scores = (query_emb * pos_emb).sum(dim=-1) # (1, )
                sum_loss = 0
                for _ in range(num_repeat):
                    # randomly sample num_negatives rows from all_negative_embeddings
                    neg_embeds = all_negative_embeddings[random.sample(range(negative_pool_size), num_negatives)].to(device)
                    negative_scores = (query_emb * neg_embeds).sum(dim=-1) # (num_neg, )
                    scores = torch.cat([positive_scores, negative_scores], dim=-1).unsqueeze(0) # (1, num_neg+1)
                    labels = torch.zeros(1, dtype=torch.long).to(device) # (1, )
            
                    loss = loss_fn(scores, labels)
                    sum_loss += loss.mean().item()
                contrast_entropy_list.append(sum_loss/num_repeat)

    return np.mean(contrast_entropy_list), contrast_entropy_list, instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_repeat", type=int, default=10)
    parser.add_argument("--negative_pool_size", type=int, default=1000000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    init_model = args.init_model

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(init_model)
    config = AutoConfig.from_pretrained(init_model)
    model = DREncoder(
        config=config,
        init_model=init_model,
        vector_dim=768,
        tokenizer=tokenizer
    ).to(device)

    info = model.load_state_dict(torch.load(init_model + "/pytorch_model.bin", map_location=device), strict=False)
    print(info)

    dataset = MSMARCO_Dataset(
        num_instances=10,
        num_negative_docs=256
    )

    set_seed(args.seed)
    loss, contrast_entropy_list, instances = contrastive_perplexity(
        model, dataset, args.negative_pool_size, args.num_repeat, device)
    print(loss)
    
    os.makedirs(pathlib.Path(args.output_path).parent, exist_ok=True)
    # dump to output_path
    with open(args.output_path, "w") as f:
        json.dump({
            "loss": loss,
            "per_query_loss": contrast_entropy_list,
            "instances": instances,
            "init_model": init_model,
            "seed": args.seed
        }, f, indent=2)


if __name__ == "__main__":
    main()
