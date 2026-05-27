"""
Modul terpisah untuk perhitungan TF-IDF dan VSM
Memastikan konsistensi dengan hasil Colab
"""
import math
from collections import defaultdict


class TFIDFCalculator:
    """
    Kelas untuk menghitung TF-IDF dengan formula yang sama seperti Colab
    Menggunakan math.log10 untuk IDF (base-10 logarithm)
    """
    
    def __init__(self, records):
        """
        Initialize TFIDF Calculator dengan records dokumen
        
        Args:
            records: List of records dengan field 'Stemming' berisi list term
        """
        self.records = records
        self.num_documents = len(records)
        self.inverted_index = None
        self.idf_scores = None
        self.tf_idf_doc_scores = None
        self.all_terms = None
    
    def build_inverted_index(self):
        """
        Membangun inverted index dari dokumen
        Memetakan setiap term ke list dokumen yang mengandungnya
        """
        self.inverted_index = defaultdict(list)
        
        for rec in self.records:
            doc_id = rec['DocID']
            stemmed_terms = rec['Stemming']
            
            for term in set(stemmed_terms):
                self.inverted_index[term].append(doc_id)
        
        # Simpan semua term yang unik
        self.all_terms = sorted(list(self.inverted_index.keys()))
        
        return self.inverted_index
    
    def calculate_idf(self):
        """
        Menghitung IDF (Inverse Document Frequency) untuk semua term
        Formula: IDF = log10(N / (df + 1))
        dimana N = total dokumen, df = document frequency (jumlah dokumen yang mengandung term)
        """
        if self.inverted_index is None:
            self.build_inverted_index()
        
        self.idf_scores = {}
        
        for term, posting_list in self.inverted_index.items():
            df = len(posting_list)  # Document Frequency
            # Formula: log10(num_docs / (df + 1)) - sama seperti Colab
            self.idf_scores[term] = math.log10(self.num_documents / (df + 1))
        
        return self.idf_scores
    
    def calculate_tf_for_document(self, stemmed_terms):
        """
        Menghitung Term Frequency (TF) untuk dokumen
        Formula: TF = frekuensi term / total terms dalam dokumen
        
        Args:
            stemmed_terms: List of stemmed terms dalam dokumen
            
        Returns:
            Dictionary dengan key=term, value=tf
        """
        tf = {}
        total_terms = len(stemmed_terms)
        
        if total_terms == 0:
            return tf
        
        term_frequency = defaultdict(int)
        for term in stemmed_terms:
            term_frequency[term] += 1
        
        for term, count in term_frequency.items():
            tf[term] = count / total_terms
        
        return tf
    
    def calculate_tf_idf(self):
        """
        Menghitung TF-IDF score untuk semua dokumen
        Formula: TF-IDF = TF * IDF
        
        Returns:
            Dictionary dengan struktur {doc_id: {term: tf_idf_score}}
        """
        if self.idf_scores is None:
            self.calculate_idf()
        
        self.tf_idf_doc_scores = {}
        
        for rec in self.records:
            doc_id = rec['DocID']
            stemmed_terms = rec['Stemming']
            
            # Hitung TF untuk dokumen ini
            tf = self.calculate_tf_for_document(stemmed_terms)
            
            # Hitung TF-IDF untuk setiap term
            tf_idf_scores = {}
            for term in set(stemmed_terms):
                tf_value = tf.get(term, 0)
                idf_value = self.idf_scores.get(term, 0)
                tf_idf_scores[term] = tf_value * idf_value
            
            self.tf_idf_doc_scores[doc_id] = tf_idf_scores
        
        return self.tf_idf_doc_scores
    
    def get_all_terms(self):
        """Mendapatkan list semua term yang unik"""
        if self.all_terms is None:
            self.build_inverted_index()
        return self.all_terms
    
    def get_idf_scores(self):
        """Mendapatkan IDF scores untuk semua term"""
        if self.idf_scores is None:
            self.calculate_idf()
        return self.idf_scores
    
    def get_tf_idf_scores(self):
        """Mendapatkan TF-IDF scores untuk semua dokumen"""
        if self.tf_idf_doc_scores is None:
            self.calculate_tf_idf()
        return self.tf_idf_doc_scores


class VSMCalculator:
    """
    Kelas untuk melakukan VSM (Vector Space Model) search
    Menggunakan TF-IDF dan Cosine Similarity untuk ranking dokumen
    """
    
    def __init__(self, tfidf_calculator):
        """
        Initialize VSM Calculator
        
        Args:
            tfidf_calculator: Instance dari TFIDFCalculator yang sudah dihitung
        """
        self.tfidf_calculator = tfidf_calculator
        self.records = tfidf_calculator.records
        self.tf_idf_doc_scores = tfidf_calculator.get_tf_idf_scores()
        self.idf_scores = tfidf_calculator.get_idf_scores()
        self.all_terms = tfidf_calculator.get_all_terms()
    
    def calculate_cosine_similarity(self, query_vector, document_vector):
        """
        Menghitung kesamaan kosinus antara dua vektor
        Formula: cosine_similarity = (A·B) / (||A|| * ||B||)
        
        Args:
            query_vector: Dictionary {term: score} untuk query
            document_vector: Dictionary {term: score} untuk dokumen
            
        Returns:
            Float antara 0-1 menyatakan similarity score
        """
        # Hitung dot product
        dot_product = sum(
            query_vector.get(term, 0) * document_vector.get(term, 0) 
            for term in set(query_vector) | set(document_vector)
        )
        
        # Hitung magnitude (norma) untuk query
        magnitude_query = math.sqrt(sum(q_val**2 for q_val in query_vector.values()))
        
        # Hitung magnitude (norma) untuk dokumen
        magnitude_document = math.sqrt(sum(d_val**2 for d_val in document_vector.values()))
        
        # Jika salah satu magnitude = 0, similarity = 0
        if magnitude_query == 0 or magnitude_document == 0:
            return 0
        
        return dot_product / (magnitude_query * magnitude_document)
    
    def process_query(self, query_text, preprocessing_func):
        """
        Memproses query text menjadi vektor TF-IDF
        
        Args:
            query_text: String query dari user
            preprocessing_func: Function untuk preprocessing query
            
        Returns:
            Dictionary {term: tf_idf_score} untuk query
        """
        # Preprocessing query
        processed_query = preprocessing_func('Query', query_text)
        query_stemmed_terms = processed_query['Stemming']
        
        if not query_stemmed_terms:
            return {}, []
        
        # Hitung TF untuk query
        query_tf = self.tfidf_calculator.calculate_tf_for_document(query_stemmed_terms)
        
        # Hitung TF-IDF untuk query
        query_tf_idf = {}
        for term in set(query_stemmed_terms):
            tf = query_tf.get(term, 0)
            idf = self.idf_scores.get(term, 0)
            query_tf_idf[term] = tf * idf
        
        return query_tf_idf, query_stemmed_terms
    
    def search(self, query_text, preprocessing_func, threshold=0):
        """
        Mencari dokumen berdasarkan kueri menggunakan VSM
        
        Args:
            query_text: String query dari user
            preprocessing_func: Function untuk preprocessing
            threshold: Minimum similarity score (0-1) untuk hasil
            
        Returns:
            List of tuples [(doc_id, similarity_score), ...] diurutkan descending
        """
        query_tf_idf, query_stemmed_terms = self.process_query(query_text, preprocessing_func)
        
        if not query_tf_idf:
            return [], []
        
        # Hitung similarity antara query dan setiap dokumen
        similarities = {}
        
        for doc_id, doc_tf_idf_vector in self.tf_idf_doc_scores.items():
            similarity = self.calculate_cosine_similarity(query_tf_idf, doc_tf_idf_vector)
            
            if similarity >= threshold:
                similarities[doc_id] = similarity
        
        # Sort dokumen berdasarkan similarity (descending)
        ranked_documents = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        
        return ranked_documents, query_stemmed_terms
    
    def get_detailed_results(self, query_text, preprocessing_func, records, threshold=0):
        """
        Mendapatkan hasil pencarian dengan detail dokumen
        
        Args:
            query_text: String query
            preprocessing_func: Function untuk preprocessing
            records: List of document records
            threshold: Minimum similarity score
            
        Returns:
            List of dictionaries dengan {doc_id, score, text_preview}
        """
        ranked_docs, query_terms = self.search(query_text, preprocessing_func, threshold)
        
        results = []
        for doc_id, score in ranked_docs:
            # Cari original text
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
