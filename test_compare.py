"""
Test script untuk compare output dengan Colab
Jalankan ini dan bandingkan hasil dengan Colab Anda
"""
import pandas as pd
from tfidf_vsm import build_inverted_index, calculate_tf_idf, query_vsm, TFIDFCalculator, VSMCalculator

# Load data Anda
def load_data():
    """Load dan preprocess data (sama dengan app.py)"""
    import re
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    stopwords = {
        'yang', 'adalah', 'ini', 'itu', 'untuk', 'dari', 'ke', 'di', 'pada', 'dengan',
        'tidak', 'juga', 'telah', 'oleh', 'dan', 'atau', 'dalam', 'bahwa', 'dapat',
        'akan', 'lebih', 'sudah', 'harus', 'ada', 'saja', 'jika', 'agar', 'atas',
        'bagi', 'antara', 'namun', 'maka', 'karena', 'sehingga', 'secara', 'seperti',
        'sesuai', 'masih', 'belum', 'bisa', 'semua', 'setiap', 'sangat', 'serta',
        'melalui', 'tetapi', 'hingga', 'saat', 'ketika', 'setelah', 'sebelum',
        'selama', 'selain', 'memiliki', 'menjadi', 'sebagai', 'suatu', 'sebuah',
        'hal', 'cara', 'lain', 'maupun', 'merupakan', 'mengenai', 'tersebut',
        'beberapa', 'berbagai', 'terhadap', 'kepada', 'tentang', 'salah', 'satu',
        'jadi', 'bukan', 'terdapat', 'berdasarkan', 'menurut', 'paling', 'hanya',
        'bahkan', 'hampir', 'pula', 'pasti', 'tentu', 'sering', 'selalu',
        'ia', 'dia', 'mereka', 'kami', 'kita', 'saya', 'anda', 'kamu',
        'dimana', 'per', 'masing', 'sendiri', 'tiap', 'sekaligus', 'kemudian',
        'iot', 'sni', 'kwh', 'rsud', 'smk', 'umkm',
    }
    
    df = pd.read_csv('TKI_Keyword_Manajemen Energi(Jurnal).csv', encoding='utf-8-sig')
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
    
    def case_folding(text):
        text = text.lower()
        text = re.sub(r'\(.*?\)', ' ', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenisasi(text):
        return [t for t in text.split() if len(t) >= 3]

    def hapus_stopword(tokens):
        return [t for t in tokens if t not in stopwords]

    def stemming(tokens):
        hasil = []
        for t in tokens:
            stem = stemmer.stem(t)
            if stem not in stopwords and len(stem) >= 3:
                hasil.append(stem)
        return hasil

    def preprocessing(doc_id, raw_text):
        folded = case_folding(raw_text)
        tokens = tokenisasi(folded)
        no_stop = hapus_stopword(tokens)
        stemmed = stemming(no_stop)
        return {
            'DocID': doc_id,
            'Teks Mentah': raw_text.strip(),
            'Case Folding': folded,
            'Tokenisasi': tokens,
            'Stopword Removal': no_stop,
            'Stemming': stemmed,
        }

    records = []
    for i, kalimat in enumerate(sentences, 1):
        record = preprocessing(f'Doc {i}', kalimat)
        records.append(record)

    return records, stemmer, stopwords


def test_tfidf():
    """Test TF-IDF calculation"""
    print("=" * 100)
    print("TEST 1: TF-IDF CALCULATION")
    print("=" * 100)
    
    records, stemmer, stopwords = load_data()
    print(f"\nTotal documents: {len(records)}")
    print(f"Sample Doc 1 terms: {records[0]['Stemming'][:10]}")
    
    # Build inverted index
    inverted_index = build_inverted_index(records)
    print(f"\nTotal unique terms: {len(inverted_index)}")
    print(f"Sample IDF for first 5 terms:")
    
    # Calculate TF-IDF
    tf_idf_doc_scores, idf_scores = calculate_tf_idf(records, inverted_index)
    
    for i, (term, idf) in enumerate(list(idf_scores.items())[:5]):
        print(f"  {term}: {idf:.6f}")
    
    # Show TF-IDF for Doc 1
    print(f"\nTF-IDF scores for Doc 1 (top 10 terms):")
    doc1_scores = sorted(tf_idf_doc_scores['Doc 1'].items(), key=lambda x: x[1], reverse=True)
    for term, score in doc1_scores[:10]:
        print(f"  {term}: {score:.6f}")
    
    return records, stemmer, stopwords


def test_vsm(records, stemmer, stopwords):
    """Test VSM search"""
    print("\n" + "=" * 100)
    print("TEST 2: VSM SEARCH")
    print("=" * 100)
    
    # Build index
    inverted_index = build_inverted_index(records)
    tf_idf_doc_scores, idf_scores = calculate_tf_idf(records, inverted_index)
    
    # Test queries
    queries = [
        "Kebutuhan energi saat ini meningkat pesat",
        "manajemen energi sistem",
        "efisiensi energi",
    ]
    
    for query in queries:
        print(f"\n[QUERY] {query}")
        ranked_docs = query_vsm(query, tf_idf_doc_scores, idf_scores, records, stemmer, stopwords)
        
        if ranked_docs:
            print(f"Found {len(ranked_docs)} documents")
            print("Top 5 results:")
            for i, (doc_id, score) in enumerate(ranked_docs[:5], 1):
                doc_text = next((r['Teks Mentah'] for r in records if r['DocID'] == doc_id), "N/A")
                print(f"  {i}. {doc_id}: {score:.6f} - {doc_text[:70]}...")
        else:
            print("No results found")


if __name__ == "__main__":
    print("\n[TEST] COMPARISON - Bandingkan dengan Colab output!\n")
    
    try:
        records, stemmer, stopwords = test_tfidf()
        test_vsm(records, stemmer, stopwords)
        
        print("\n" + "=" * 100)
        print("[DONE] Test selesai! Bandingkan output di atas dengan output Colab Anda")
        print("=" * 100)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
