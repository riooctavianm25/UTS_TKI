"""
Modul terpisah untuk perhitungan TF-IDF dan VSM
Struktur EXACT seperti Colab - tidak ada perubahan logic
"""
import math
import re
from collections import defaultdict


def build_inverted_index(records):
    """
    Membangun inverted index dari records
    Format: {term: [doc_id1, doc_id2, ...]}
    """
    inverted_index = defaultdict(list)
    
    for rec in records:
        doc_id = rec['DocID']
        stemmed_terms = rec['Stemming']
        
        # Setiap term unique dalam dokumen hanya ditambahkan 1x
        for term in set(stemmed_terms):
            inverted_index[term].append(doc_id)
    
    return inverted_index


def calculate_tf_idf(records, inverted_index):
    """
    Menghitung bobot TF-IDF untuk setiap term di setiap dokumen.
    EXACT seperti Colab - tanpa perubahan.
    
    Args:
        records: List of records dengan 'DocID' dan 'Stemming'
        inverted_index: Dictionary mapping term ke list doc_ids
        
    Returns:
        tf_idf_scores: {doc_id: {term: tf_idf_score}}
        idf_scores: {term: idf_value}
    """
    tf_idf_scores = defaultdict(dict)
    num_documents = len(records)

    # Menghitung IDF (Inverse Document Frequency)
    idf_scores = {}
    for term, posting_list in inverted_index.items():
        df = len(posting_list)  # Frekuensi Dokumen
        idf_scores[term] = math.log10(num_documents / (df + 1))

    # Menghitung TF-IDF
    for record in records:
        doc_id = record['DocID']
        stemmed_terms = record['Stemming']
        term_freq = defaultdict(int)
        for term in stemmed_terms:
            term_freq[term] += 1

        for term in set(stemmed_terms):
            raw_tf = term_freq[term]
            tf = 1 + math.log10(raw_tf)  # Log frequency weighting
            idf = idf_scores.get(term, 0)
            tf_idf_scores[doc_id][term] = tf * idf

    return tf_idf_scores, idf_scores


def calculate_cosine_similarity(query_vector, document_vector):
    """
    Menghitung kesamaan kosinus antara dua vektor.
    EXACT seperti Colab.
    """
    dot_product = sum(
        query_vector.get(term, 0) * document_vector.get(term, 0) 
        for term in set(query_vector) | set(document_vector)
    )

    magnitude_query = math.sqrt(sum(q_val**2 for q_val in query_vector.values()))
    magnitude_document = math.sqrt(sum(d_val**2 for d_val in document_vector.values()))

    if magnitude_query == 0 or magnitude_document == 0:
        return 0
    return dot_product / (magnitude_query * magnitude_document)


def query_vsm(query_text, tf_idf_doc_scores, idf_scores, records, stemmer, stopwords):
    """
    Mencari dokumen berdasarkan kueri menggunakan VSM dan kesamaan kosinus.
    EXACT seperti Colab.
    
    Args:
        query_text: String query
        tf_idf_doc_scores: {doc_id: {term: score}} dari calculate_tf_idf
        idf_scores: {term: idf} dari calculate_tf_idf
        records: List of records
        stemmer: PySastrawi stemmer instance
        stopwords: Set of stopwords
        
    Returns:
        ranked_documents: [(doc_id, similarity_score), ...] sorted by score DESC
    """
    # 1. Preprocessing kueri (inline, sama seperti Colab)
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

    # Preprocessing query
    folded = case_folding(query_text)
    tokens = tokenisasi(folded)
    no_stop = hapus_stopword(tokens)
    query_stemmed_terms = stemming(no_stop)

    if not query_stemmed_terms:
        return []

    # 2. Menghitung TF untuk kueri
    query_tf = defaultdict(int)
    for term in query_stemmed_terms:
        query_tf[term] += 1

    # 3. Menghitung TF-IDF untuk kueri
    query_tf_idf = {}
    if len(query_stemmed_terms) > 0:
        for term in set(query_stemmed_terms):
            tf = query_tf[term] / len(query_stemmed_terms)
            idf = idf_scores.get(term, 0)
            query_tf_idf[term] = tf * idf

    # 4. Menghitung kesamaan kosinus antara kueri dan setiap dokumen
    similarities = {}
    for doc_id, doc_tf_idf_vector in tf_idf_doc_scores.items():
        similarity = calculate_cosine_similarity(query_tf_idf, doc_tf_idf_vector)
        if similarity > 0:
            similarities[doc_id] = similarity

    # 5. Mengurutkan dokumen berdasarkan kesamaan
    ranked_documents = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

    return ranked_documents


# ============================================================================
# CLASS-BASED INTERFACE (untuk backward compatibility dengan app.py)
# ============================================================================

class TFIDFCalculator:
    """
    Wrapper class untuk calculate_tf_idf function
    Memudahkan penggunaan di app.py
    """
    
    def __init__(self, records):
        self.records = records
        self.num_documents = len(records)
        self.inverted_index = None
        self.idf_scores = None
        self.tf_idf_doc_scores = None
        self.all_terms = None
    
    def build_inverted_index(self):
        self.inverted_index = build_inverted_index(self.records)
        self.all_terms = sorted(list(self.inverted_index.keys()))
        return self.inverted_index
    
    def calculate_idf(self):
        if self.inverted_index is None:
            self.build_inverted_index()
        
        self.idf_scores = {}
        for term, posting_list in self.inverted_index.items():
            df = len(posting_list)
            self.idf_scores[term] = math.log10(self.num_documents / (df + 1))
        
        return self.idf_scores
    
    def calculate_tf_for_document(self, stemmed_terms):
        """Calculate TF for a document using log frequency weighting"""
        tf = {}
        
        if len(stemmed_terms) == 0:
            return tf
        
        term_frequency = defaultdict(int)
        for term in stemmed_terms:
            term_frequency[term] += 1
        
        for term, count in term_frequency.items():
            tf[term] = 1 + math.log10(count)  # Log frequency weighting
        
        return tf
    
    def calculate_tf_idf(self):
        """Calculate TF-IDF for all documents"""
        if self.inverted_index is None:
            self.build_inverted_index()
        
        self.tf_idf_doc_scores, self.idf_scores = calculate_tf_idf(self.records, self.inverted_index)
        return self.tf_idf_doc_scores
    
    def get_all_terms(self):
        if self.all_terms is None:
            self.build_inverted_index()
        return self.all_terms
    
    def get_idf_scores(self):
        if self.idf_scores is None:
            self.calculate_idf()
        return self.idf_scores
    
    def get_tf_idf_scores(self):
        if self.tf_idf_doc_scores is None:
            self.calculate_tf_idf()
        return self.tf_idf_doc_scores


class VSMCalculator:
    """
    Wrapper class untuk query_vsm function
    Memudahkan penggunaan di app.py
    """
    
    def __init__(self, tfidf_calculator, stemmer, stopwords):
        self.tfidf_calculator = tfidf_calculator
        self.records = tfidf_calculator.records
        self.tf_idf_doc_scores = tfidf_calculator.get_tf_idf_scores()
        self.idf_scores = tfidf_calculator.get_idf_scores()
        self.stemmer = stemmer
        self.stopwords = stopwords
    
    def calculate_cosine_similarity(self, query_vector, document_vector):
        """Wrapper untuk calculate_cosine_similarity function"""
        return calculate_cosine_similarity(query_vector, document_vector)
    
    def search(self, query_text, threshold=0):
        """Search documents based on query - EXACT seperti Colab"""
        ranked_docs = query_vsm(query_text, self.tf_idf_doc_scores, self.idf_scores, 
                               self.records, self.stemmer, self.stopwords)
        
        # Filter by threshold
        filtered_docs = [(doc_id, score) for doc_id, score in ranked_docs if score >= threshold]
        
        return filtered_docs
    
    def get_detailed_results(self, query_text, records, threshold=0):
        """Get detailed search results"""
        # Search menggunakan query_vsm langsung
        ranked_docs = query_vsm(query_text, self.tf_idf_doc_scores, self.idf_scores, 
                               self.records, self.stemmer, self.stopwords)
        
        # Filter by threshold
        filtered_docs = [(doc_id, score) for doc_id, score in ranked_docs if score >= threshold]
        
        # Extract query terms
        def case_folding(text):
            text = text.lower()
            text = re.sub(r'\(.*?\)', ' ', text)
            text = re.sub(r'[^a-z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def tokenisasi(text):
            return [t for t in text.split() if len(t) >= 3]

        def hapus_stopword(tokens):
            return [t for t in tokens if t not in self.stopwords]

        def stemming(tokens):
            hasil = []
            for t in tokens:
                stem = self.stemmer.stem(t)
                if stem not in self.stopwords and len(stem) >= 3:
                    hasil.append(stem)
            return hasil
        
        folded = case_folding(query_text)
        tokens = tokenisasi(folded)
        no_stop = hapus_stopword(tokens)
        query_terms = stemming(no_stop)
        
        results = []
        for doc_id, score in filtered_docs:
            original_text = next(
                (rec['Teks Mentah'] for rec in records if rec['DocID'] == doc_id),
                'N/A'
            )
            
            results.append({
                'DocID': doc_id,
                'Similarity Score': score,
                'Text Preview': original_text[:100] + '...' if len(original_text) > 100 else original_text,
                'Full Text': original_text
            })
        
        return results, query_terms


