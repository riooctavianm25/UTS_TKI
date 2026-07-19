"""
Hybrid Retrieval: menggabungkan TF-IDF lexical dan Dense semantic untuk hasil lebih baik
"""
import pandas as pd
import numpy as np
from tfidf_vsm import TFIDFCalculator, VSMCalculator
from dense_retrieval import DenseRetrievalEngine
from evaluation import load_qrels, evaluate_system
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

from app import preprocess_tfidf

# Load data
df = pd.read_csv("TKI_Keyword_Manajemen Energi(Jurnal).csv")
stemmer = StemmerFactory().create_stemmer()

# Extract sentences
sentences = []
for index, row in df.iterrows():
    line = str(row.iloc[0]).strip()
    if not line or pd.isna(line) or line == 'nan':
        continue
    if line.startswith('http'):
        continue
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    line = line.strip()
    if len(line) > 10:
        sentences.append(line)

# Create records
stopwords = set()
records = []
for i, text in enumerate(sentences[:50], 1):
    tfidf_result = preprocess_tfidf(text, stemmer, stopwords)
    records.append({
        'DocID': f'Doc {i}',
        'Teks Mentah': text,
        'Case Folding': tfidf_result['Case Folding'],
        'Tokenisasi': tfidf_result['Tokenisasi'],
        'Stopword Removal': tfidf_result['Stopword Removal'],
        'Stemming': tfidf_result['Stemming'],
        'SBERT Clean Text': text,
    })

# Initialize both systems
print("Initializing TF-IDF...")
tfidf = TFIDFCalculator(records)
tfidf.calculate_tf_idf()
stopwords = set()
vsm = VSMCalculator(tfidf, stemmer, stopwords)

print("Initializing Dense Retrieval...")
dense = DenseRetrievalEngine(records)
dense.load_bi_encoder()
dense.encode_documents()
dense.build_faiss_index()

# Load queries
queries_df = pd.read_csv("queries.csv")
queries = queries_df["query_text"].tolist()

# Load ground truth
qrels_graded, qrels_binary = load_qrels('qrels.csv')

# Hybrid search function
def hybrid_search(query_text, qid, top_k=10, alpha=0.5):
    """
    Combine TF-IDF and Dense retrieval results
    alpha: weight for dense results (1-alpha for TF-IDF)
    """
    # TF-IDF search
    tfidf_results = vsm.search(query_text)  # Get all results, then slice
    tfidf_results = tfidf_results[:20]
    tfidf_ranking = {result[0]: (1 / (i + 1)) for i, result in enumerate(tfidf_results)}  # RRF score
    
    # Dense search  
    dense_results = dense.search(query_text, top_k=20)
    dense_ranking = {result.doc_id: (1 / (i + 1)) for i, result in enumerate(dense_results)}  # RRF score
    
    # Combine scores
    all_docs = set(list(tfidf_ranking.keys()) + list(dense_ranking.keys()))
    combined_scores = {}
    for doc_id in all_docs:
        tfidf_score = tfidf_ranking.get(doc_id, 0)
        dense_score = dense_ranking.get(doc_id, 0)
        # Weighted combination
        combined_scores[doc_id] = (1 - alpha) * tfidf_score + alpha * dense_score
    
    # Sort by combined score
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# Test hybrid search
print("\n" + "="*80)
print("Testing Hybrid Retrieval (alpha=0.5, equal weight)")
print("="*80)

for qid in ['1', '2', '3']:
    query_text = queries[int(qid) - 1]
    print(f"\nQuery {qid}: {query_text[:70]}...")
    print(f"Ground truth: {sorted(qrels_binary[qid])}")
    
    results = hybrid_search(query_text, qid, alpha=0.5)
    print("Hybrid results (top 10):")
    for i, (doc_id, score) in enumerate(results[:10]):
        match = ' [R]' if doc_id in qrels_binary[qid] else ''
        print(f"  {i+1}. {doc_id} (score={score:.4f}){match}")

# Full evaluation with hybrid
print("\n" + "="*80)
print("Full Hybrid Evaluation")
print("="*80)

hybrid_results = {}
for qid_int, qid_str in enumerate(range(1, 6), 1):
    qid = str(qid_str)
    query_text = queries[qid_int - 1]
    results = hybrid_search(query_text, qid, alpha=0.5, top_k=10)
    hybrid_results[qid] = [doc_id for doc_id, _ in results]

# Evaluate
try:
    metrics = evaluate_system('Hybrid', hybrid_results, qrels_graded, qrels_binary)
    print(f"Hybrid MAP: {metrics['map']:.4f}")
    print(f"Hybrid MRR: {metrics['mrr']:.4f}")
    print(f"Hybrid NDCG: {metrics['ndcg']:.4f}")
except Exception as e:
    print(f"Evaluation error: {e}")
