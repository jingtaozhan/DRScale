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

import faiss
from tqdm import tqdm
import json
import torch

import pytrec_eval
from collections import defaultdict
from typing import Dict, Union
from dataset import tokenize
from transformers.trainer_utils import set_seed


def truncate_run(run: Dict[str, Dict[str, float]], topk: int):
    new_run = dict()
    for qid, pid2scores in run.items():
        rank_lst = sorted(pid2scores.items(), key=lambda x: x[1], reverse=True)
        new_run[qid] = dict(rank_lst[:topk])
    return new_run


def pytrec_evaluate(
        qrel: Union[str, Dict[str, Dict[str, int]]], 
        run: Union[str, Dict[str, float]],
        k_values =(1, 3, 5, 10, 100, 1000),
        mrr_k_values = (10, 100, 1000),
        recall_k_values = (10, 50, 100, 200, 500, 1000),
        relevance_level = 1,
        ):
    qrel = {
        str(qid): {str(pid): rel for pid, rel in pid2rel.items()} for qid, pid2rel in qrel.items()
    }
    run = {
        str(qid): {str(pid): score for pid, score in pid2score.items()} for qid, pid2score in run.items()
    }
    ndcg, map, recall, precision, mrr = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in recall_k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    if isinstance(qrel, str):
        with open(qrel, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)
    if isinstance(run, str):
        with open(run, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {map_string, ndcg_string, recall_string, precision_string}, relevance_level=relevance_level)
    query_scores = evaluator.evaluate(run)
    
    for query_id in query_scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += query_scores[query_id]["ndcg_cut_" + str(k)]
            map[f"MAP@{k}"] += query_scores[query_id]["map_cut_" + str(k)]
            precision[f"P@{k}"] += query_scores[query_id]["P_"+ str(k)]
        for k in recall_k_values:
            recall[f"Recall@{k}"] += query_scores[query_id]["recall_" + str(k)]
    
    if len(query_scores) < len(qrel):
        missing_qids = qrel.keys() - query_scores.keys()
        print(f"Missing results for {len(missing_qids)} queries!")
        for query_id in missing_qids:
            query_scores[query_id] = dict()
            for k in k_values:
                query_scores[query_id]["ndcg_cut_" + str(k)] = 0
                query_scores[query_id]["map_cut_" + str(k)] = 0
                query_scores[query_id]["P_"+ str(k)] = 0
            for k in recall_k_values:
                query_scores[query_id]["recall_" + str(k)] = 0

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(query_scores), 5)
        map[f"MAP@{k}"] = round(map[f"MAP@{k}"]/len(query_scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(query_scores), 5)
    for k in recall_k_values:
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(query_scores), 5)

    mrr_evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {"recip_rank"}, relevance_level=relevance_level)
    for mrr_cut in mrr_k_values:
        mrr_query_scores = mrr_evaluator.evaluate(truncate_run(run, mrr_cut))
        for query_id in mrr_query_scores.keys():
            s = mrr_query_scores[query_id]["recip_rank"]
            mrr[f"MRR@{mrr_cut}"] += s
            query_scores[query_id][f"recip_rank_{mrr_cut}"] = s
        mrr[f"MRR@{mrr_cut}"] = round(mrr[f"MRR@{mrr_cut}"]/len(mrr_query_scores), 5)

    ndcg, map, recall, precision, mrr = dict(ndcg), dict(map), dict(recall), dict(precision), dict(mrr)
    metric_scores = {
        "ndcg": ndcg,
        "map": map,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "perquery": query_scores
    }
    return dict(metric_scores)


def encode_corpus(model: DREncoder, dataset: MSMARCO_Dataset, negative_ids, device: torch.device) -> np.ndarray:
    all_embeddings = np.zeros((len(negative_ids), 768), dtype=np.float32)
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
            all_embeddings[start_idx:end_idx] = embeds.detach().cpu().numpy()
    return all_embeddings


def run_eval(model: DREncoder, dataset: MSMARCO_Dataset, pool_size: int, device: torch.device, device_index:int):
    num_negatives = dataset.num_negative_docs
    eval_batch_size = 64

    model.eval()

    queries = dataset.test_pairs['human'][0]
    qrels = dataset.test_pairs['human'][1]
    corpus = dataset.corpus
    tokenizer = model.tokenizer

    valid_qids = [qid for qid in queries.keys() if len(qrels[qid]) > 0]
    maybe_positive_doc_ids = set()
    for qid in valid_qids:
        maybe_positive_doc_ids |= set(qrels[qid].keys())

    all_doc_ids = set(corpus.keys())
    negative_doc_ids = sorted(list(all_doc_ids - maybe_positive_doc_ids))
    pool_doc_ids = list(maybe_positive_doc_ids) + random.sample(negative_doc_ids, pool_size - len(maybe_positive_doc_ids))
    all_embeddings = encode_corpus(
        model, dataset, pool_doc_ids, device
    )
    # construct a IP index with all_embeddings using faiss
    index = faiss.IndexFlatIP(768)
    index.add(all_embeddings)
    # put the index to one single gpu (the same as device)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, device_index, index)

    assert len(valid_qids) > 0

    instances = [(qid, pid) for qid in valid_qids for pid in qrels[qid]]

    # print info
    print(f"Total instances: {len(instances)}")
    print(f"Total valid queries: {len(valid_qids)}")
    print(f"Total negative docs: {len(negative_doc_ids)}")
    print(f"Total positive docs: {len(maybe_positive_doc_ids)}")
    print(f"Num sample_negatives: {num_negatives}")

    search_ranks_results = dict()
    for start_idx in tqdm(range(0, len(valid_qids), eval_batch_size), desc="Contrastive perplexity"):
        end_idx = min(start_idx + eval_batch_size, len(instances))
        batch_qids = valid_qids[start_idx:end_idx]

        batch_queries = [queries[qid] for qid in batch_qids]
        batch_queries = tokenize(batch_queries, tokenizer, 32)

        with torch.no_grad():
            query_embes = model.batch_encode(
                batch_queries['input_ids'].to(device),
                batch_queries['attention_mask'].to(device),
                8
            ).detach().cpu().numpy()
            # search the index to get the positive embeddings using faiss
            batch_sims, batch_pidxes = index.search(query_embes, 1000)
            for qid, sims, pidxes in zip(batch_qids, batch_sims, batch_pidxes):
                pids = [pool_doc_ids[i] for i in pidxes.tolist()]
                sims = sims.tolist()
                search_ranks_results[qid] = dict(zip(pids, sims))
    
    metrics = pytrec_evaluate(qrels, search_ranks_results)
    
    return {
        "metric": metrics,
        "rank": search_ranks_results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--metric_output_path", type=str, required=True)
    parser.add_argument("--rank_output_path", type=str, required=True)
    parser.add_argument("--pool_size", type=int, default=1000000)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    init_model = args.init_model

    device = torch.device(f"cuda:{args.device}")
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
    results = run_eval(
        model, dataset, args.pool_size, device, device_index=args.device)
    print({k: results['metric'][k] for k in results['metric'] if k != "perquery"})
    
    os.makedirs(pathlib.Path(args.metric_output_path).parent, exist_ok=True)
    with open(args.metric_output_path, 'w') as f:
        json.dump(results['metric'], f, indent=2)
    
    os.makedirs(pathlib.Path(args.rank_output_path).parent, exist_ok=True)
    with open(args.rank_output_path, 'w') as f:
        json.dump(results['rank'], f, indent=2)


if __name__ == "__main__":
    main()
