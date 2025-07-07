import torch
import random, json
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
import json
import gzip
import os
import tarfile
from typing import List, Dict

from utils import *


class DRDataset(Dataset):
    def __init__(self, 
            queries: Dict[str, str],                  # query_id -> query_text
            corpus: Dict[str, str],                   # doc_id -> doc_text
            qrels: Dict[str, List[str]],              # query_id -> [relevant_doc_id]
            num_instances: int=-1,                    # number of training instances
            num_negative_per_query: int=512           # number of negative documents per query
        ) -> None:
        
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels
        self.num_instances = num_instances
        self.num_negative_docs = num_negative_per_query

        self.corpus_ids = list(self.corpus.keys())

        self.instances = []
        for qid in self.qrels:
            for pid in self.qrels[qid]:
                self.instances.append((qid, pid))

        # random shuffle
        random.shuffle(self.instances)

        if self.num_instances > 0:
            self.instances = self.instances[:self.num_instances]


        print("-" * 40)
        print("#> Build DR Dataset")
        print("#>  - num_queries", len(self.queries))
        print("#>  - num_documents", len(self.corpus_ids))
        print("#>  - num_negative_docs", self.num_negative_docs)
        print("#>  - num_qrels", len(self.qrels))
        print("#>  - average num positive docs", np.mean([len(v) for v in self.qrels.values()]))
        print("#>  - num_instances", len(self.instances))
        print("-" * 40)

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        qid, pid = self.instances[idx]
        query_text = self.queries[qid]
        pos_text = self.corpus[pid]

        return {
            "qid": qid,
            "query": query_text,
            "pos_id": pid,
            "pos": pos_text
        }

class MSMARCO_Dataset(DRDataset):
    def __init__(self, 
            num_instances: int=-1,
            num_negative_docs: int=512
        ) -> None:
        corpus, queries, qrels = load_msmarco()
        qrel = load_msmarco_pos(list(queries.keys()), corpus, qrels)
        super().__init__(queries, corpus, qrel, num_instances, num_negative_docs)

        test_queries = load_msmarco_queries("dev.small")
        test_qrels = load_msmarco_qrels("dev.small")

        self.test_pairs = {
            "human": (test_queries, test_qrels),
        }


class GLM_MSMARCO_NQ1D_Dataset(DRDataset):
    def __init__(self, nq1d="1q1d", num_instances=-1, num_negative_docs=512):
        corpus = load_msmarco_corpus()
        queries = load_glm_queries("train")
        qrels = load_glm_qrels(f"train.{nq1d}")

        qrel = load_msmarco_pos(list(queries.keys()), corpus, qrels)
        super().__init__(queries, corpus, qrel, num_instances, num_negative_docs)

        test_queries = load_msmarco_queries("dev.small")
        test_qrels = load_msmarco_qrels("dev.small")

        test_gpt_queries = load_glm_queries("test")
        test_gpt_qrels = load_glm_qrels("test")

        self.test_pairs = {
            "human": (test_queries, test_qrels),
            # "glm": (test_gpt_queries, test_gpt_qrels),
        }


class T5_MSMARCO_Dataset(DRDataset):
    def __init__(self, num_instances=-1, num_negative_docs=512):
        corpus = load_msmarco_corpus()
        # queries = load_glm_queries("train")
        # qrels = load_glm_qrels(f"train.{nq1d}")
        queries, qrels = load_t5_queries_qrels()

        qrel = load_msmarco_pos(list(queries.keys()), corpus, qrels)
        super().__init__(queries, corpus, qrel, num_instances, num_negative_docs)

        test_queries = load_msmarco_queries("dev.small")
        test_qrels = load_msmarco_qrels("dev.small")

        self.test_pairs = {
            "human": (test_queries, test_qrels),
            # "glm": (test_gpt_queries, test_gpt_qrels),
        }


class ICT_MSMARCO_Dataset(DRDataset):
    def __init__(self, num_instances=-1, num_negative_docs=512):
        corpus = load_msmarco_corpus()
        # queries = load_glm_queries("train")
        # qrels = load_glm_qrels(f"train.{nq1d}")
        queries, qrels, ict_corpus = load_ict_queries_qrels(corpus)
        self.ict_corpus = ict_corpus
        qrel = load_msmarco_pos(list(queries.keys()), corpus, qrels)
        super().__init__(queries, corpus, qrel, num_instances, num_negative_docs)

        test_queries = load_msmarco_queries("dev.small")
        test_qrels = load_msmarco_qrels("dev.small")

        self.test_pairs = {
            "human": (test_queries, test_qrels),
            # "glm": (test_gpt_queries, test_gpt_qrels),
        }
    
    def __getitem__(self, idx):
        qid, pid = self.instances[idx]
        query = self.queries[qid]
        pos = self.ict_corpus[pid]
        return {
            "qid": qid,
            "query": query,
            "pos_id": pid,
            "pos": pos
        }


def tokenize(texts, tokenizer, max_input_length):
    # texts = [t[:1000] for t in texts]
    features = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=max_input_length)
    tensor_features = {
        'input_ids': torch.LongTensor(features['input_ids']),
        'attention_mask': torch.LongTensor(features['attention_mask']),
    }

    return tensor_features

def get_collate_function(max_query_length, max_doc_length, tokenizer, dataset):

    def collate_function(batch):
        queries = [item["query"] for item in batch]
        pos_ids = [item["pos_id"] for item in batch]

        pos_ids_set = set(pos_ids)

        # sample negative docs
        num_negative_docs = dataset.num_negative_docs
        sample_num = num_negative_docs + len(pos_ids_set) # sample more to ensure we have enough negative docs

        neg_ids = random.sample(dataset.corpus_ids, sample_num)
        neg_ids = [nid for nid in neg_ids if nid not in pos_ids_set] # remove positive docs
        neg_ids = neg_ids[:num_negative_docs]


        pos = [item["pos"] for item in batch]
        neg = [dataset.corpus[nid] for nid in neg_ids]

        query_features = tokenize(queries, tokenizer, max_query_length)
        pos_features = tokenize(pos, tokenizer, max_doc_length)
        neg_features = tokenize(neg, tokenizer, max_doc_length)

        data = {
            "query": query_features,
            "pos": pos_features,
            "neg": neg_features
        }
        return data

    return collate_function  


if __name__ == "__main__":
    ds = MSMARCO_Dataset(
        num_instances=10,
        num_negative_docs=5
    )

    
