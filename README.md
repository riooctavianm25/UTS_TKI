# Information Retrieval - Manajemen Energi

Aplikasi Information Retrieval untuk dokumen jurnal tentang Manajemen Energi dengan dukungan pencarian berbasis kata kunci, pencarian semantik, dan evaluasi hasil secara terstruktur.

## Deskripsi Project

Proyek ini mengimplementasikan sistem pencarian informasi yang mencakup beberapa pendekatan:

- Pencarian berbasis lexical menggunakan TF-IDF dan VSM.
- Pencarian semantik menggunakan embedding SBERT dan FAISS.
- Pencarian hybrid yang menggabungkan hasil lexical dan semantic.
- Evaluasi performa sistem menggunakan metrik MAP, MRR, dan NDCG.

Tujuan utama aplikasi ini adalah membantu pengguna mencari dokumen yang relevan dari kumpulan data jurnal terkait Manajemen Energi secara cepat dan terukur.

---

## Fitur Utama

- Preprocessing teks otomatis:
  - case folding
  - tokenisasi
  - stopword removal
  - stemming menggunakan Sastrawi

- Pencarian multi-mode:
  - Boolean Search
  - VSM Search
  - Semantic Search
  - Hybrid Retrieval

- Dense retrieval:
  - pemanfaatan model SBERT/bi-encoder
  - dukungan FAISS untuk pencarian vektor cepat
  - opsi reranking dengan cross-encoder jika tersedia

- Evaluasi sistem:
  - memuat query dari file queries.csv
  - memuat ground truth dari file qrels.csv
  - menghasilkan metrik evaluasi MAP, MRR, dan NDCG

---

## Struktur Project

```text
TKI/
├── app.py                          # Aplikasi Streamlit utama
├── tfidf_vsm.py                   # Implementasi TF-IDF, VSM, dan semantic search
├── dense_retrieval.py             # Mesin dense retrieval berbasis SBERT + FAISS
├── hybrid_retrieval.py            # Script pencarian hybrid dan evaluasi
├── evaluation.py                  # Modul evaluasi MAP/MRR/NDCG
├── preprocessing_utils.py        # Utilitas preprocessing
├── preprocessing_manajemen_energi_&_tf_idf_vsm.ipynb
├── README.md
├── queries.csv                    # Daftar query uji
├── qrels.csv                      # Ground truth relevansi dokumen
├── TKI_Keyword_Manajemen Energi(Jurnal).csv
├── TKI_Keyword_Manajemen Energi(Jurnal)_50.csv
└── hasil_evaluasi.csv            # Hasil evaluasi jika dieksekusi
```

---

## Requirements

Pastikan Anda menggunakan Python 3.9+ atau 3.10+ dan menginstal dependency berikut:

```bash
pip install streamlit pandas numpy Sastrawi sentence-transformers faiss-cpu
```

Jika ingin memanfaatkan reranking cross-encoder, sistem juga akan mencoba memuat paket yang relevan dari Hugging Face secara otomatis saat tersedia.

---

## Instalasi

### 1. Clone / masuk ke folder project

```bash
cd "l:\Semester 6\TKI"
```

### 2. Buat virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependency

```bash
pip install streamlit pandas numpy Sastrawi sentence-transformers faiss-cpu
```

---

## Cara Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser default pada alamat:

```text
http://localhost:8501
```

### Menu yang tersedia di aplikasi

- Shelf Monitoring
  - Preprocessing
  - Inverted Index
  - Forward Index

- Search Engine
  - Boolean Search
  - VSM Search
  - Semantic Search
  - Evaluation

---

## Alur Kerja Sistem

### 1. Loading Data

Data dokumen dibaca dari file CSV jurnal yang berisi teks terkait Manajemen Energi.

### 2. Preprocessing

Teks diproses melalui tahapan:

```text
Case Folding -> Tokenisasi -> Stopword Removal -> Stemming
```

### 3. Indexing dan Pencarian

Sistem membangun representasi dokumen untuk dua pendekatan:

- lexical: TF-IDF + VSM
- semantic: embedding dense + FAISS

### 4. Ranking dan Evaluasi

Hasil pencarian kemudian dirangking dan dapat dievaluasi menggunakan query dan qrels yang tersedia.

---

## Evaluasi

File yang digunakan untuk evaluasi:

- queries.csv: daftar query uji
- qrels.csv: relevansi dokumen terhadap query

Untuk menjalankan evaluasi hybrid, gunakan:

```bash
python hybrid_retrieval.py
```

Hasil evaluasi akan disimpan ke file:

```text
hasil_evaluasi.csv
```

---

## Catatan Penting

- Jika dependency dense retrieval belum terinstall, aplikasi tetap dapat berjalan untuk mode TF-IDF/VSM dengan pemberitahuan warning.
- Untuk performa terbaik, disarankan menginstal dependency dense retrieval lengkap.
- Proyek ini masih terus berkembang, sehingga dokumentasi dapat disesuaikan seiring penambahan fitur baru.

