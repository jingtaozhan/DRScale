import sys
sys.path.append("./")
import torch
import os
import random, json
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import pickle
import json
import gzip
import os
import tarfile

def file_tqdm(file):
    print(f"#> Reading {file.name}")

    with tqdm(total=os.path.getsize(file.name) / 1024.0 / 1024.0, unit="MiB") as pbar:
        for line in file:
            yield line
            pbar.update(len(line) / 1024.0 / 1024.0)

        pbar.close()

msmarco_corpus_path = '/home/fangyan/Workspace/ColBERT/data/dataset/collection.tsv'
msmarco_queries_path = '/home/fangyan/Workspace/ColBERT/data/dataset/queries.train.tsv'


def load_msmarco_corpus():
    collection_filepath = os.path.join('/home/fangyan/Workspace/ColBERT/data/dataset', 'collection.tsv')
    corpus = {}
    print("Read corpus: collection.tsv")
    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in file_tqdm(fIn):
            pid, passage = line.strip().split("\t")
            pid = int(pid)
            corpus[pid] = passage
    return corpus

def load_msmarco_queries(split='train'):
    queries_filepath = os.path.join('/home/fangyan/Workspace/ColBERT/data/dataset', f'queries.{split}.tsv')
    queries = {}
    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in file_tqdm(fIn):
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query
    return queries


def load_msmarco_qrels(split="train"):
    qrels_filepath = os.path.join('/home/fangyan/Workspace/ColBERT/data/dataset', f'qrels.{split}.tsv')
    qrels = defaultdict(dict)
    with open(qrels_filepath, 'r', encoding='utf8') as fIn:
        for line in file_tqdm(fIn):
            qid, _, pid, label = map(int, line.strip().split())
            assert label == 1
            qrels[qid][pid] = label
    return qrels


def load_msmarco_posneg(qids, corpus, qrels):
    qid2posneg = {}
    corpus_ids = list(corpus.keys())
    print("#> Sampling positive and negative passages for each query")
    for qid in tqdm(qids):
        pos_pids = list(qrels[qid].keys())
        neg_pids = random.sample(corpus_ids, 100)

        # remove positive pids from negative pids
        neg_pids = [pid for pid in neg_pids if pid not in pos_pids]

        if len(pos_pids) > 0 and len(neg_pids) > 10:
            qid2posneg[qid] = {'pos': [], 'neg': []}
            qid2posneg[qid]['pos'] = pos_pids
            qid2posneg[qid]['neg'] = neg_pids

            random.shuffle(qid2posneg[qid]['pos'])
            random.shuffle(qid2posneg[qid]['neg'])

    return qid2posneg


def load_msmarco_posneg_with_constrain(qids, corpus, qrels, pos_constrain, neg_constrain):
    qid2posneg = {}
    if neg_constrain is None:
        corpus_ids = list(corpus.keys())
    else:
        corpus_ids = neg_constrain

    
    print("#> Sampling positive and negative passages for each query with constrain")
    
    if pos_constrain is not None:
        pos_constrain = set(pos_constrain)
        print("#> Pos constrain:", len(pos_constrain))
    if neg_constrain is not None:
        print("#> Neg constrain:", len(neg_constrain))
    
    for qid in tqdm(qids):
        pos_pids = list(qrels[qid].keys())
        neg_pids = random.sample(corpus_ids, 100)

        if pos_constrain is not None:
            if not any([pid in pos_constrain for pid in pos_pids]):
                continue
        # remove positive pids from negative pids
        neg_pids = [pid for pid in neg_pids if pid not in pos_pids]

        if len(pos_pids) > 0 and len(neg_pids) > 10:
            qid2posneg[qid] = {'pos': [], 'neg': []}
            qid2posneg[qid]['pos'] = pos_pids
            qid2posneg[qid]['neg'] = neg_pids

            random.shuffle(qid2posneg[qid]['pos'])
            random.shuffle(qid2posneg[qid]['neg'])

    return qid2posneg


def load_msmarco_pos(qids, corpus, qrels):
    qid2pos = {}
    corpus_ids = list(corpus.keys())
    print("#> Sampling only positive passages for each query")
    for qid in tqdm(qids):
        pos_pids = list(qrels[qid].keys())
        if len(pos_pids) > 0:
            qid2pos[qid] = pos_pids
            random.shuffle(qid2pos[qid])

    return qid2pos


def load_gpt_queries(split='train'):
    queries = {}
    with open(f"/home/fangyan/Workspace/scaling_law/datasets/gpt.queries.{split}.tsv", 'r', encoding='utf8') as fIn:
        for line in file_tqdm(fIn):
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query
    return queries

def load_gpt_qrels(split='train'):
    qrels_filepath = f"/home/fangyan/Workspace/scaling_law/datasets/gpt.qrels.{split}.tsv"
    qrels = defaultdict(dict)
    with open(qrels_filepath, 'r', encoding='utf8') as fIn:
        for line in file_tqdm(fIn):
            qid, _, pid, label = map(int, line.strip().split())
            assert label == 1
            qrels[qid][pid] = label
    return qrels


def load_glm_queries(split='train'):
    queries = {}
    with open(f"/home/fangyan/Workspace/scaling_law/datasets/glm.queries.{split}.tsv", 'r', encoding='utf8') as fIn:
        for line in file_tqdm(fIn):
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query
    return queries

def load_glm_qrels(split='train'):
    qrels_filepath = f"/home/fangyan/Workspace/scaling_law/datasets/glm.qrels.{split}.tsv"
    qrels = defaultdict(dict)
    with open(qrels_filepath, 'r', encoding='utf8') as fIn:
        for line in file_tqdm(fIn):
            qid, _, pid, label = map(int, line.strip().split())
            assert label == 1
            qrels[qid][pid] = label
    return qrels

def load_t5_queries_qrels(split='train'):
    filepath = "/home/fangyan/Workspace/datasets/t5/d2q.jsonl.gz"
    qrels = defaultdict(dict)
    queries = {}
    max_passages = 1000000
    with gzip.open(filepath, "rt") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            # print(line["id"], len(line["predicted_queries"]), line["predicted_queries"][:3])
            pid = int(line["id"])
            qid = i + 1
            query = line["predicted_queries"][0]
            if i >= max_passages:
                break
            queries[pid] = query
            qrels[qid][pid] = 1

    return queries, qrels
            
from nltk.tokenize import sent_tokenize
def load_ict_queries_qrels(corpus, split='train'):
    
    qrels = defaultdict(dict)
    queries = {}
    ict_corpus = {}
    max_passages = 1000000
    qid = 0
    for pid, passage in corpus.items():
        qid += 1
        
        sentences = sent_tokenize(passage)
        ### sample two spans
        sentences = [s for s in sentences if len(s.split(' ')) >= 3]

        if len(sentences) < 2:
            query = " ".join(passage.split(' ')[:20])
            pos = " ".join(passage.split(' ')[-20:])
        else:
            if random.random() < 0.5:
                query = sentences[0]
                pos = " ".join(sentences[1:])
            else:
                query_idx = random.sample(range(len(sentences)), 1)[0]
                query = sentences[query_idx]
                pos = " ".join(sentences[:query_idx] + sentences[query_idx+1:])
        
        if qid >= max_passages:
            break
        queries[pid] = query
        qrels[qid][pid] = 1
        ict_corpus[pid] = pos

    return queries, qrels, ict_corpus
            


def load_t2ranking_queries(split='train'):
    queries_filepath = f"/home/fangyan/Dataset/T2Ranking/data/queries.{split}.tsv"
    queries = {}
    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in file_tqdm(fIn):
            # skip the first line
            qid, query = line.strip().split("\t")
            if qid == "qid":
                continue
            qid = int(qid)
            queries[qid] = query
    return queries

def load_t2ranking_corpus():
    collection_filepath = "/home/fangyan/Dataset/T2Ranking/data/collection.tsv"
    corpus = {}
    print("Read corpus: collection.tsv")
    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in file_tqdm(fIn):
            pid, passage = line.strip().split("\t")
            if pid == "pid":
                continue
            pid = int(pid)
            corpus[pid] = passage
    return corpus

def load_t2ranking_qrels(split='train'):
    qrels_filepath = f"/home/fangyan/Dataset/T2Ranking/data/qrels.retrieval.{split}.tsv"
    qrels = defaultdict(dict)
    with open(qrels_filepath, 'r', encoding='utf8') as fIn:
        for line in file_tqdm(fIn):
            qid, pid = line.strip().split("\t")
            if qid == "qid":
                continue
            qid = int(qid)
            pid = int(pid)
            qrels[qid][pid] = 1
    return qrels



def load_t2ranking_pos(qids, corpus, qrels):
    qid2pos = {}
    corpus_ids = list(corpus.keys())
    print("#> Sampling only positive passages for each query")
    for qid in tqdm(qids):
        pos_pids = list(qrels[qid].keys())
        if len(pos_pids) > 0:
            qid2pos[qid] = pos_pids
            random.shuffle(qid2pos[qid])

    return qid2pos

def load_t2ranking():
    corpus = load_t2ranking_corpus()
    queries = load_t2ranking_queries()
    qrels = load_t2ranking_qrels()
    return corpus, queries, qrels

def load_msmarco_hardneg(qids, corpus, qrels):
    hard_negatives_filepath = os.path.join('/home/fangyan/Workspace/ColBERT/data/dataset', 'msmarco-hard-negatives.jsonl.gz')
    print("Read hard negatives train file")
    train_queries = {}
    negs_to_use = None
    # negs_to_use = ["bm25"]
    num_negs_per_system = 10
    max_passages = 0
    with gzip.open(hard_negatives_filepath, 'rt') as fIn:
        for line in tqdm(fIn):
            if max_passages > 0 and len(train_queries) >= max_passages:
                break
            data = json.loads(line)

            #Get the positive passage ids
            pos_pids = data['pos']

            #Get the hard negatives
            neg_pids = set()
            if negs_to_use is None:
                negs_to_use = list(data['neg'].keys())
                print("Using negatives from the following systems:", negs_to_use)

            for system_name in negs_to_use:
                if system_name not in data['neg']:
                    continue

                system_negs = data['neg'][system_name]
                negs_added = 0
                for pid in system_negs:
                    if pid not in neg_pids:
                        neg_pids.add(pid)
                        negs_added += 1
                        if negs_added >= num_negs_per_system:
                            break
            
            if len(pos_pids) > 0 and len(neg_pids) > 10:
                neg_pids = list(neg_pids)
                train_queries[data['qid']] = {'qid': data['qid'], 'pos': pos_pids, 'neg': neg_pids}

    return train_queries


def load_msmarco():
    corpus = load_msmarco_corpus()
    queries = load_msmarco_queries()
    qrels = load_msmarco_qrels()
    return corpus, queries, qrels


def load_lotte_corpus(dataset):
    corpus = {}
    with open(f'../datasets/downloads/lotte/{dataset}/test/collection.tsv') as f:
        for line in file_tqdm(f):
            doc_id, text = line.strip().split('\t')
            corpus[doc_id] = text
    return corpus

def load_lotte_queries(dataset, split='forum'):
    queries = {}
    with open(f'../datasets/downloads/lotte/{dataset}/test/questions.{split}.tsv') as f:
        for line in file_tqdm(f):
            qid, query = line.strip().split('\t')
            queries[qid] = query
    return queries




def load_lotte_qrels(dataset, split='forum'):
    qrels = {}
    with open(f'../datasets/downloads/lotte/{dataset}/test/qas.{split}.jsonl') as f:
        for line in file_tqdm(f):
            line = json.loads(line)
            qid = str(line["qid"])
            pos_pids = [str(pid) for pid in line["answer_pids"]]
            qrels[qid] = {pid: 1 for pid in pos_pids}
    return qrels

def load_lotte_test(dataset):
    corpus = load_lotte_corpus(dataset)
    queries = load_lotte_queries(dataset)
    qrels = load_lotte_qrels(dataset)
    return corpus, queries, qrels



def load_lotte_sentence_queries(dataset, split='forum'):
    queries = {}
    with open(f'../datasets/downloads/lotte/{dataset}/test/sentence_topic.tsv') as f:
        for line in file_tqdm(f):
            qid, query = line.strip().split('\t')
            queries[qid] = query
    return queries

def load_lotte_bm25_topk(dataset, max_line=-1):
    qid2posneg = {}
    line_cnt = 0
    with open(f'../datasets/downloads/lotte/{dataset}/test/sentence_bm25_top200.trec') as f:
        for line in file_tqdm(f):
            qid, _, doc_id, rank, score, _ = line.strip().split(' ')
            rank = int(rank)

            if qid not in qid2posneg:
                qid2posneg[qid] = {'pos': [], 'neg': []}

            ### top10 as pos, top50-100 as neg
            if rank <= 10:
                qid2posneg[qid]['pos'].append(doc_id)
            elif rank > 50:
                qid2posneg[qid]['neg'].append(doc_id)

            # random shuffle
            random.shuffle(qid2posneg[qid]['pos'])
            random.shuffle(qid2posneg[qid]['neg'])

            line_cnt += 1
            if max_line > 0 and line_cnt >= max_line:
                break

    return qid2posneg


def load_lotte_bm25_top10(dataset, corpus, max_line=-1):
    qid2pos = {}
    line_cnt = 0
    with open(f'../datasets/downloads/lotte/{dataset}/test/sentence_bm25_top10.trec') as f:
        for line in file_tqdm(f):
            qid, _, doc_id, rank, score, _ = line.strip().split(' ')
            rank = int(rank)

            if qid not in qid2pos:
                qid2pos[qid] = []

            if dataset == "msmarco":
                doc_id = int(doc_id)
            ### top10 as pos, top50-100 as neg
            if rank <= 3 and doc_id in corpus:
                qid2pos[qid].append(doc_id)
            
            # random shuffle
            random.shuffle(qid2pos[qid])

            line_cnt += 1
            if max_line > 0 and line_cnt >= max_line:
                break
        
        ### keep only qid with at least 5 pos
        qid2pos = {qid: pos for qid, pos in qid2pos.items() if len(pos) >= 1}

    return qid2pos




def load_lotte_bm25(dataset):
    corpus = load_lotte_corpus(dataset)
    queries = load_lotte_sentence_queries(dataset)
    qid2posneg = load_lotte_bm25_topk(dataset)
    return corpus, queries, qid2posneg


def load_weak_beir(dataset):
    queries = {}
    qrels = defaultdict(dict)
    with open(f'../datasets/downloads/beir/{dataset}/qrel.weak.jsonl') as f:
        for line in file_tqdm(f):
            line = json.loads(line)
            qid = str(line["qid"])
            queries[qid] = line['query']
            qrels[line['qid']][line['pos_id']] = 1
    return queries, qrels


def load_beir_queries_with_pos_text(dataset):
    qid2postext = defaultdict(list)
    with open(f'../datasets/downloads/beir/{dataset}/qrel.weak.jsonl') as f:
        for line in file_tqdm(f):
            line = json.loads(line)
            qid = str(line["qid"])

            qid2postext[qid].append(line['pos_text'])
            
    return qid2postext

def load_beir_corpus(dataset):
    collection_filepath = f'/home/fangyan/Workspace/DenseRetrieval/datasets/{dataset}/corpus.jsonl'
    corpus = {}
    print("Read corpus: collection.tsv")
    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in file_tqdm(fIn):
            line = json.loads(line)
            pid = str(line["_id"])
            if "title" in line and line["title"] != "":
                text = line["title"] + ". " + line["text"]
            else:
                text = line["text"]
            # if len(text.split(' ')) > 5:
            corpus[pid] = text
    return corpus

def load_beir_queries(dataset, split='train'):
    queries = {}
    with open(f'/home/fangyan/Workspace/DenseRetrieval/datasets/{dataset}/queries.jsonl') as f:
        for line in file_tqdm(f):
            line = json.loads(line)
            qid = str(line["_id"])
            queries[qid] = line['text']
    return queries

def load_beir_qrels(dataset, split='train'):
    qrels = defaultdict(dict)
    with open(f'/home/fangyan/Workspace/DenseRetrieval/datasets/{dataset}/qrels/{split}.tsv') as f:
        for line in file_tqdm(f):
            if line.startswith("query"): continue
            qid, pid, score = line.strip().split('\t')
            qid = str(qid)
            pid = str(pid)
            score = int(score)
            qrels[qid][pid] = score
    return qrels

def load_beir_posneg(dataset, qids, corpus, qrels):
    qid2posneg = {}
    corpus_ids = list(corpus.keys())
    print("#> Sampling positive and negative passages for each query")
    for qid in tqdm(qids):
        pos_pids = list(qrels[qid].keys())


        if pos_pids[0].startswith("MED"):
            neg_pids = random.sample(corpus_ids[:3633], 100)
        else:
            neg_pids = random.sample(corpus_ids[3633:], 100)

        # remove positive pids from negative pids
        neg_pids = [pid for pid in neg_pids if pid not in pos_pids]

        if len(pos_pids) > 0 and len(neg_pids) > 20:
            qid2posneg[qid] = {'pos': [], 'neg': []}
            qid2posneg[qid]['pos'] = pos_pids
            qid2posneg[qid]['neg'] = neg_pids

            random.shuffle(qid2posneg[qid]['pos'])
            random.shuffle(qid2posneg[qid]['neg'])

    return qid2posneg


def load_beir_hard_posneg(dataset, qids, corpus, qrels):
    qid2posneg = {}
    corpus_ids = list(corpus.keys())
    print("#> Sampling positive and negative passages for each query")

    hard_negs = {}
    with open(f'../datasets/downloads/beir/{dataset}/hard_negatives.jsonl') as f:
        for line in file_tqdm(f):
            line = json.loads(line)
            qid = str(line["qid"])
            hard_negs[qid] = line["neg"][10:100]

    for qid in tqdm(qids):
        pos_pids = list(qrels[qid].keys())
        if qid in hard_negs:
            neg_pids = random.sample(hard_negs[qid], 20)
        else:
            neg_pids = random.sample(corpus_ids, 20)

        # remove positive pids from negative pids
        neg_pids = [pid for pid in neg_pids if pid not in pos_pids]

        if len(pos_pids) > 0 and len(neg_pids) > 5:
            qid2posneg[qid] = {'pos': [], 'neg': []}
            qid2posneg[qid]['pos'] = pos_pids
            qid2posneg[qid]['neg'] = neg_pids

            random.shuffle(qid2posneg[qid]['pos'])
            random.shuffle(qid2posneg[qid]['neg'])

    return qid2posneg


def load_lotte_qgen_queries(dataset):
    queries = {}
    with open(f'/home/zjt/lotte/{dataset}/test/query.gen') as f:
        for line in file_tqdm(f):
            try:
                qid, query = line.strip().split('\t')
                queries[qid] = query
            except:
                continue
    return queries

def load_lotte_qgen_qrels(dataset, corpus):
    qid2pos = {}
    line_cnt = 0
    with open(f'/home/zjt/lotte/{dataset}/test/qrels.gen') as f:
        for line in file_tqdm(f):
            qid, _, doc_id, score = line.strip().split('\t')
            
            if qid not in qid2pos:
                qid2pos[qid] = []

            if doc_id in corpus:
                qid2pos[qid].append(doc_id)
            
            random.shuffle(qid2pos[qid])

            line_cnt += 1
    qid2pos = {qid: pos for qid, pos in qid2pos.items() if len(pos) >= 1}
    return qid2pos


def load_beir_qgen_queries_and_qrels(dataset, corpus):
    from datasets import load_dataset
    dataset = load_dataset(f"BeIR/{dataset}-generated-queries")
    qid = 0
    qid2query = {}
    qid2pos = {}
    for data in tqdm(dataset['train']):
        if data["_id"] not in corpus:
            continue
        qid2query[str(qid)] = data['query']

        qid2pos[str(qid)] = [data['_id']]
        qid += 1

    qid2pos = {qid: pos for qid, pos in qid2pos.items() if len(pos) >= 1}
    
    print(f"#> Loaded {len(qid2query)} queries and {len(qid2pos)} qrels")
    return qid2query, qid2pos


def tokenize(texts, tokenizer, max_input_length):
    texts = [t[:10000] for t in texts]
    features = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=max_input_length)
    tensor_features = {
        'input_ids':torch.LongTensor(features['input_ids']),
        'attention_mask': torch.LongTensor(features['attention_mask'])
    }
    return tensor_features




def make_prefix(domain_id, task_id, num_token=1):
    prefix = ""
    for i in range(num_token):
        domain_token = f"[DOMAIN{domain_id}-{i}] "
        prefix += domain_token
    
    for i in range(num_token):
        task_token = f"[TASK{task_id}-{i}] "
        prefix += task_token

    return prefix
