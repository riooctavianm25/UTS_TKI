"""Utilitas Dense Retrieval menggunakan SBERT bi-encoder dan Cross-Encoder reranking.

Pipeline legacy TF-IDF/VSM tidak diubah. Modul ini dimuat secara lazy
sehingga aplikasi tetap berjalan meskipun dependensi dense retrieval belum terpasang.
"""

import re
from dataclasses import dataclass


# Coba muat sentence-transformers; jika tidak ada, set ke None agar modul tetap bisa diimport
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:  # pragma: no cover - dependensi opsional
    SentenceTransformer = None
    CrossEncoder = None


# Coba muat faiss; jika tidak ada, set ke None agar modul tetap bisa diimport
try:
    import faiss
except Exception:  # pragma: no cover - dependensi opsional
    faiss = None


@dataclass
class DenseSearchResult:
    """Representasi satu hasil pencarian dari pipeline dense retrieval."""
    doc_id: str       # Identifikasi dokumen
    score: float      # Skor relevansi (cosine similarity atau skor cross-encoder)
    text_preview: str # Cuplikan teks 160 karakter pertama
    full_text: str    # Teks lengkap dokumen


class DenseRetrievalEngine:
    """
    Mesin pencarian dense retrieval dengan dua tahap:
    1. Bi-Encoder (SBERT + FAISS) untuk retrieval cepat (Tahap 1)
    2. Cross-Encoder untuk reranking presisi tinggi (Tahap 2)
    """

    def __init__(
        self,
        records,
        text_field="SBERT Clean Text",
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ):
        """
        Inisialisasi mesin retrieval.

        Args:
            records: Daftar dict dokumen yang sudah dipreproses
            text_field: Nama kolom teks yang dipakai untuk embedding SBERT
            model_name: ID model bi-encoder dari HuggingFace Hub
        """
        self.records = records
        self.text_field = text_field
        self.model_name = model_name
        self.model = None       # Bi-encoder (dimuat secara lazy)
        self.reranker = None    # Cross-encoder (dimuat secara lazy)
        self.embeddings = None  # Matriks embedding dokumen (50 × 768)
        self.index = None       # FAISS IndexFlatIP
        # Ambil teks untuk setiap dokumen saat inisialisasi
        self.doc_texts = [self._get_doc_text(rec) for rec in records]

    def _clean_text(self, text):
        """Jaga teks tetap natural untuk SBERT/BERT: tidak stemming, tidak hapus stopword, hanya normalisasi ringan."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _get_doc_text(self, record):
        """
        Ambil teks dokumen dengan prioritas kolom:
        SBERT Clean Text > Case Folding > Teks Mentah
        """
        raw_text = (
            record.get(self.text_field)
            or record.get("Case Folding")
            or record.get("Teks Mentah", "")
        )
        return self._clean_text(raw_text)

    def _require_sentence_transformers(self):
        """Periksa apakah sentence-transformers sudah terpasang, lempar error jika belum."""
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers belum terpasang. "
                "Jalankan: pip install sentence-transformers"
            )

    def _require_faiss(self):
        """Periksa apakah faiss sudah terpasang, lempar error jika belum."""
        if faiss is None:
            raise ImportError(
                "faiss belum terpasang. "
                "Jalankan: pip install faiss-cpu"
            )

    def load_bi_encoder(self):
        """
        Muat model bi-encoder SBERT dari HuggingFace Hub (lazy loading).
        Model di-cache ke self.model agar tidak dimuat ulang.
        """
        self._require_sentence_transformers()
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def load_reranker(self, reranker_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        """
        Muat model cross-encoder untuk reranking (lazy loading).
        Model di-cache ke self.reranker agar tidak dimuat ulang.

        Args:
            reranker_name: ID model cross-encoder dari HuggingFace Hub
        """
        self._require_sentence_transformers()
        if self.reranker is None:
            self.reranker = CrossEncoder(reranker_name)
        return self.reranker

    def encode_documents(self):
        """
        Encode seluruh dokumen menjadi vektor embedding dense menggunakan SBERT (Tahap Offline).
        Normalisasi L2 diterapkan agar inner product setara dengan cosine similarity.

        Returns:
            embeddings: Matriks numpy float32 berukuran (jumlah_dokumen × 768)
        """
        model = self.load_bi_encoder()
        self.embeddings = model.encode(
            self.doc_texts,
            batch_size=16,           # Ukuran batch untuk inferensi bertahap
            show_progress_bar=False, # Nonaktifkan progress bar di Streamlit
            normalize_embeddings=True,  # Normalisasi L2 otomatis
        )
        # Pastikan dtype float32 yang dibutuhkan FAISS
        self.embeddings = self.embeddings.astype("float32")
        return self.embeddings

    def build_faiss_index(self):
        """
        Bangun FAISS IndexFlatIP dari embedding dokumen (Tahap Offline).
        IndexFlatIP = Inner Product, setara cosine similarity setelah normalisasi L2.

        Returns:
            index: Objek FAISS index yang siap digunakan untuk pencarian
        """
        self._require_faiss()
        if self.embeddings is None:
            # Encode dokumen terlebih dahulu jika belum ada
            self.encode_documents()
        dimension = self.embeddings.shape[1]  # Dimensi embedding = 768
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)       # Tambahkan semua vektor dokumen ke index
        return self.index

    def search(self, query_text, top_k=10):
        """
        Cari top-k dokumen paling relevan menggunakan bi-encoder + FAISS (Tahap Online).

        Args:
            query_text: Teks query dari pengguna
            top_k: Jumlah kandidat yang diambil dari FAISS

        Returns:
            results: Daftar DenseSearchResult yang sudah diurutkan berdasarkan skor cosine
        """
        # Bangun index jika belum ada
        if self.index is None:
            self.build_faiss_index()

        # Encode query menjadi vektor embedding
        model = self.load_bi_encoder()
        cleaned_query = self._clean_text(query_text)
        query_embedding = model.encode([cleaned_query], normalize_embeddings=True)
        query_embedding = query_embedding.astype("float32")

        # Cari top-k kandidat di FAISS
        scores, indices = self.index.search(query_embedding, top_k)

        # Konversi hasil FAISS ke objek DenseSearchResult
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                # Index -1 berarti tidak ada kandidat (terjadi jika k > jumlah dokumen)
                continue
            record = self.records[idx]
            full_text = record.get("Teks Mentah", self.doc_texts[idx])
            # Buat cuplikan 160 karakter pertama
            preview = full_text[:160] + ("..." if len(full_text) > 160 else "")
            results.append(
                DenseSearchResult(record["DocID"], float(score), preview, full_text)
            )
        return results

    def rerank(self, query_text, candidates, reranker_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        """
        Reranking kandidat menggunakan Cross-Encoder untuk presisi lebih tinggi (Tahap Online).
        Cross-Encoder menilai setiap pasangan (query, dokumen) secara bersama.

        Args:
            query_text: Teks query dari pengguna
            candidates: Daftar DenseSearchResult dari tahap retrieval bi-encoder
            reranker_name: ID model cross-encoder dari HuggingFace Hub

        Returns:
            reranked: Daftar DenseSearchResult yang sudah diurutkan ulang berdasarkan skor cross-encoder
        """
        reranker = self.load_reranker(reranker_name)
        # Buat pasangan (query, teks_dokumen) untuk setiap kandidat
        pairs = [(query_text, candidate.full_text) for candidate in candidates]
        # Prediksi skor relevansi untuk setiap pasangan
        ce_scores = reranker.predict(pairs)
        # Urutkan dari skor tertinggi ke terendah
        sorted_pairs = sorted(
            zip(candidates, ce_scores), key=lambda item: item[1], reverse=True
        )

        reranked = []
        for candidate, score in sorted_pairs:
            reranked.append(
                DenseSearchResult(
                    candidate.doc_id,
                    float(score),
                    candidate.text_preview,
                    candidate.full_text,
                )
            )
        return reranked
