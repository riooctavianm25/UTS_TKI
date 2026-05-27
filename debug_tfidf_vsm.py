"""
Debug file untuk membandingkan perhitungan TF-IDF dan VSM dengan Colab
Trace setiap step untuk menemukan perbedaan
"""
import math
from collections import defaultdict
from tfidf_vsm import TFIDFCalculator, VSMCalculator

def debug_tfidf_calculation(records, inverted_index_colab=None):
    """
    Debug TF-IDF calculation dan compare dengan Colab
    """
    print("=" * 80)
    print("DEBUG: TF-IDF CALCULATION")
    print("=" * 80)
    
    tfidf_calc = TFIDFCalculator(records)
    
    # 1. Build Inverted Index
    print("\n1. BUILDING INVERTED INDEX")
    print("-" * 80)
    inv_idx = tfidf_calc.build_inverted_index()
    print(f"Total unique terms: {len(inv_idx)}")
    print(f"Sample terms (first 5):")
    for term in list(inv_idx.keys())[:5]:
        print(f"  {term}: {inv_idx[term]}")
    
    # 2. Calculate IDF
    print("\n2. CALCULATING IDF")
    print("-" * 80)
    idf_scores = tfidf_calc.calculate_idf()
    print(f"Sample IDF scores (first 5 terms):")
    for term in list(idf_scores.keys())[:5]:
        df = len(inv_idx[term])
        formula = f"log10({tfidf_calc.num_documents} / ({df} + 1))"
        value = idf_scores[term]
        print(f"  {term}: {value:.6f} [{formula}]")
    
    # 3. Calculate TF-IDF
    print("\n3. CALCULATING TF-IDF FOR DOCUMENTS")
    print("-" * 80)
    tf_idf_docs = tfidf_calc.calculate_tf_idf()
    
    # Show TF-IDF for first document
    first_doc_id = records[0]['DocID']
    print(f"\nTF-IDF scores for {first_doc_id}:")
    first_doc_tfidf = tf_idf_docs[first_doc_id]
    sorted_terms = sorted(first_doc_tfidf.items(), key=lambda x: x[1], reverse=True)
    print(f"Top 5 terms:")
    for term, score in sorted_terms[:5]:
        doc_stemmed = records[0]['Stemming']
        tf = doc_stemmed.count(term) / len(doc_stemmed) if doc_stemmed else 0
        idf = idf_scores[term]
        formula = f"TF({tf:.4f}) × IDF({idf:.4f})"
        print(f"  {term}: {score:.6f} [{formula}]")
    
    return tfidf_calc, inv_idx, idf_scores, tf_idf_docs


def debug_vsm_search(tfidf_calc, records, query_text, preprocessing_func, stemmer, stopwords):
    """
    Debug VSM search dan trace setiap step
    """
    print("\n" + "=" * 80)
    print("DEBUG: VSM SEARCH")
    print("=" * 80)
    
    vsm_calc = VSMCalculator(tfidf_calc)
    
    # 1. Preprocessing query
    print("\n1. QUERY PREPROCESSING")
    print("-" * 80)
    processed_query = preprocessing_func('Query', query_text)
    query_stemmed = processed_query['Stemming']
    print(f"Original query: {query_text}")
    print(f"Stemmed terms: {query_stemmed}")
    print(f"Unique terms: {len(set(query_stemmed))}")
    
    # 2. Calculate Query TF
    print("\n2. QUERY TERM FREQUENCY (TF)")
    print("-" * 80)
    query_tf_dict = defaultdict(int)
    for term in query_stemmed:
        query_tf_dict[term] += 1
    
    print(f"Raw term frequencies:")
    for term, count in sorted(query_tf_dict.items(), key=lambda x: x[1], reverse=True):
        tf_normalized = count / len(query_stemmed)
        print(f"  {term}: {count} → TF = {count}/{len(query_stemmed)} = {tf_normalized:.4f}")
    
    # 3. Calculate Query TF-IDF
    print("\n3. QUERY TF-IDF")
    print("-" * 80)
    query_tf_idf, _ = vsm_calc.process_query(query_text, preprocessing_func)
    print(f"Query TF-IDF vector (top terms):")
    sorted_query = sorted(query_tf_idf.items(), key=lambda x: x[1], reverse=True)
    for term, score in sorted_query[:5]:
        print(f"  {term}: {score:.6f}")
    
    # Query vector magnitude
    query_magnitude = math.sqrt(sum(v**2 for v in query_tf_idf.values()))
    print(f"\nQuery vector magnitude: {query_magnitude:.6f}")
    
    # 4. Calculate Similarity for sample documents
    print("\n4. COSINE SIMILARITY FOR SAMPLE DOCUMENTS")
    print("-" * 80)
    
    idf_scores = tfidf_calc.get_idf_scores()
    tf_idf_docs = tfidf_calc.get_tf_idf_scores()
    
    # Test first 3 documents
    for i in range(min(3, len(records))):
        doc_id = records[i]['DocID']
        doc_tfidf = tf_idf_docs[doc_id]
        
        # Calculate similarity
        dot_prod = sum(
            query_tf_idf.get(term, 0) * doc_tfidf.get(term, 0)
            for term in set(query_tf_idf) | set(doc_tfidf)
        )
        
        doc_magnitude = math.sqrt(sum(v**2 for v in doc_tfidf.values()))
        
        if query_magnitude == 0 or doc_magnitude == 0:
            similarity = 0
        else:
            similarity = dot_prod / (query_magnitude * doc_magnitude)
        
        print(f"\n{doc_id}:")
        print(f"  Dot product: {dot_prod:.6f}")
        print(f"  Doc magnitude: {doc_magnitude:.6f}")
        print(f"  Cosine similarity: {similarity:.6f}")
    
    # 5. Full ranking
    print("\n5. FULL RANKING")
    print("-" * 80)
    ranked_docs, _ = vsm_calc.search(query_text, preprocessing_func, threshold=0)
    print(f"Total documents ranked: {len(ranked_docs)}")
    print(f"Top 10 results:")
    for i, (doc_id, score) in enumerate(ranked_docs[:10], 1):
        print(f"  {i}. {doc_id}: {score:.6f}")


def create_test_comparison():
    """
    Buat comparison test dengan sample data kecil
    """
    print("\n" + "=" * 80)
    print("TEST COMPARISON WITH SAMPLE DATA")
    print("=" * 80)
    
    # Create minimal sample records
    sample_records = [
        {
            'DocID': 'Doc 1',
            'Teks Mentah': 'energi solar adalah energi terbarukan',
            'Stemming': ['energi', 'solar', 'energi', 'terbaruken']
        },
        {
            'DocID': 'Doc 2',
            'Teks Mentah': 'panel solar menggunakan energi matahari',
            'Stemming': ['panel', 'solar', 'guna', 'energi', 'matahari']
        },
        {
            'DocID': 'Doc 3',
            'Teks Mentah': 'efisiensi energi dalam industri',
            'Stemming': ['efisien', 'energi', 'industri']
        }
    ]
    
    print("\nSample records:")
    for rec in sample_records:
        print(f"  {rec['DocID']}: {rec['Stemming']}")
    
    # Calculate TF-IDF
    tfidf_calc = TFIDFCalculator(sample_records)
    tfidf_calc.build_inverted_index()
    tfidf_calc.calculate_idf()
    tfidf_calc.calculate_tf_idf()
    
    print("\nIDF scores:")
    for term, idf in sorted(tfidf_calc.idf_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {term}: {idf:.4f}")
    
    print("\nTF-IDF for each document:")
    for doc_id, terms_scores in tfidf_calc.tf_idf_doc_scores.items():
        print(f"\n{doc_id}:")
        for term, score in sorted(terms_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {term}: {score:.6f}")


if __name__ == "__main__":
    # Untuk dijalankan manual saat debugging
    print("Debug module untuk TF-IDF dan VSM calculation")
    print("Gunakan: debug_tfidf_calculation(records) dan debug_vsm_search(...)")
