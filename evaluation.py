"""
Modul Evaluasi Sistem IR (Information Retrieval).

Mengimplementasikan metrik evaluasi standar IR sesuai PRD Bab 8:
- MAP@k  (Mean Average Precision)
- MRR@k  (Mean Reciprocal Rank)
- NDCG@k (Normalized Discounted Cumulative Gain)

Modul ini digunakan oleh app.py untuk mengevaluasi dan membandingkan
tiga sistem secara nyata: TF-IDF Baseline, SBERT-Only, dan SBERT + Cross-Encoder.
"""

import math
import os
import re
from collections import defaultdict

import pandas as pd


# ============================================================================
# FUNGSI METRIK EVALUASI
# ============================================================================

def _normalize_doc_id(doc_id):
    if doc_id is None:
        return ""

    doc_id = str(doc_id).strip()
    if not doc_id:
        return ""

    match = re.match(r"^(?:doc\s*)?(\d+)$", doc_id, flags=re.IGNORECASE)
    if match:
        return f"Doc {int(match.group(1))}"

    return doc_id


def compute_precision_at_k(ranked_list, relevant_docs_binary, k):
    if k == 0 or not ranked_list:
        return 0.0
    top_k = ranked_list[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_docs_binary)
    return hits / k


def compute_ap(ranked_list, relevant_docs_binary, k=10):
    if not relevant_docs_binary:
        return 0.0
    total_relevant = len(relevant_docs_binary)
    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(ranked_list[:k], start=1):
        if doc_id in relevant_docs_binary:
            hits += 1
            sum_precision += hits / i
    return sum_precision / total_relevant


def compute_map(ap_scores):
    if not ap_scores:
        return 0.0
    return sum(ap_scores) / len(ap_scores)


def compute_rr(ranked_list, relevant_docs_binary, k=10):
    for i, doc_id in enumerate(ranked_list[:k], start=1):
        if doc_id in relevant_docs_binary:
            return 1.0 / i
    return 0.0


def compute_mrr(rr_scores):
    if not rr_scores:
        return 0.0
    return sum(rr_scores) / len(rr_scores)


def compute_dcg(ranked_list, graded_relevance, k=10):
    dcg = 0.0
    for i, doc_id in enumerate(ranked_list[:k], start=1):
        rel = graded_relevance.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def compute_ndcg(ranked_list, graded_relevance, k=10):
    dcg = compute_dcg(ranked_list, graded_relevance, k)
    ideal_ranking = sorted(
        graded_relevance.keys(),
        key=lambda d: graded_relevance[d],
        reverse=True
    )
    idcg = compute_dcg(ideal_ranking, graded_relevance, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def load_qrels(qrels_path="qrels.csv"):
    if not os.path.exists(qrels_path):
        return None, None

    df = pd.read_csv(qrels_path)
    qrels_graded = defaultdict(dict)
    qrels_binary = defaultdict(set)

    for _, row in df.iterrows():
        query_id = str(row['query_id']).strip()
        doc_id = _normalize_doc_id(row['doc_id'])
        score = int(row['relevance_score'])

        qrels_graded[query_id][doc_id] = score
        if score >= 1:
            qrels_binary[query_id].add(doc_id)

    return dict(qrels_graded), dict(qrels_binary)


def evaluate_system(system_name, ranked_results_per_query, qrels_graded, qrels_binary, k=10):
    ap_scores = []
    rr_scores = []
    ndcg_scores = []
    per_query_results = []

    available_qids = set()
    if isinstance(qrels_binary, dict):
        available_qids.update(qrels_binary.keys())
    if isinstance(qrels_graded, dict):
        available_qids.update(qrels_graded.keys())

    eval_qids = [qid for qid in ranked_results_per_query.keys() if qid in available_qids] if available_qids else list(ranked_results_per_query.keys())

    for query_id in eval_qids:
        ranked_list = [_normalize_doc_id(doc_id) for doc_id in ranked_results_per_query.get(query_id, [])]
        rel_binary = {_normalize_doc_id(doc_id) for doc_id in qrels_binary.get(query_id, set())}
        rel_graded = {_normalize_doc_id(doc_id): score for doc_id, score in qrels_graded.get(query_id, {}).items()}

        ap = compute_ap(ranked_list, rel_binary, k)
        rr = compute_rr(ranked_list, rel_binary, k)
        ndcg = compute_ndcg(ranked_list, rel_graded, k)

        ap_scores.append(ap)
        rr_scores.append(rr)
        ndcg_scores.append(ndcg)
        per_query_results.append({'query_id': query_id, 'ap': round(ap, 4), 'rr': round(rr, 4), 'ndcg': round(ndcg, 4)})

    if not per_query_results:
        return {'system': system_name, 'map': 0.0, 'mrr': 0.0, 'ndcg': 0.0, 'per_query': []}

    return {
        'system': system_name,
        'map': round(compute_map(ap_scores), 4),
        'mrr': round(compute_mrr(rr_scores), 4),
        'ndcg': round(sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0, 4),
        'per_query': per_query_results,
    }


def save_results_csv(all_results, output_path="hasil_evaluasi.csv"):
    rows = []
    for result in all_results:
        system = result['system']
        rows.append({'Sistem': system, 'Query': 'RATA-RATA', 'MAP': result['map'], 'MRR': result['mrr'], 'NDCG': result['ndcg']})
        for pq in result['per_query']:
            rows.append({'Sistem': system, 'Query': pq['query_id'], 'MAP': pq['ap'], 'MRR': pq['rr'], 'NDCG': pq['ndcg']})
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return output_path


def get_summary_table(all_results, k=10):
    rows = []
    for result in all_results:
        rows.append({
            'Sistem': result['system'],
            f'MAP@{k}': result['map'],
            f'MRR@{k}': result['mrr'],
            f'NDCG@{k}': result['ndcg'],
        })
    return pd.DataFrame(rows)


def get_per_query_table(all_results, metric='map'):
    rows = []
    for result in all_results:
        for pq in result['per_query']:
            rows.append({'Query': pq['query_id'], 'Sistem': result['system'], 'Skor': pq.get(metric, 0)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pivot = df.pivot(index='Query', columns='Sistem', values='Skor').reset_index()
    return pivot

