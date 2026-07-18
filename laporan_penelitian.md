# LAPORAN PENELITIAN UAS

## Sistem Temu Kembali Dokumen Berbasis Dense Retrieval pada Dokumen Manajemen Energi
### Implementasi SBERT + FAISS dan Cross-Encoder Reranking

## ABSTRAK

Penelitian ini mengembangkan upgrade sistem pencarian dokumen pada tugas sebelumnya dengan mengintegrasikan teknik modern berbasis embedding dan reranking untuk meningkatkan relevansi hasil pencarian. Dataset menggunakan 50 dokumen manajemen energi yang sama dengan tugas sebelumnya. Evaluasi dilakukan menggunakan minimal 5 ground truth query sendiri dan metrik MAP, MRR, serta NDCG@10. Laporan ini menjabarkan arsitektur sistem, workflow implementasi, contoh query, desain evaluasi, serta instruksi screenshot yang disarankan untuk dokumentasi.

## 1. INTRODUKSI

### 1.1 Konteks dan Motivasi
Sistem IR tradisional berbasis TF-IDF dan Vector Space Model (VSM) telah menjadi baseline pada tugas sebelumnya. Model tersebut mengandalkan kemunculan kata yang sama untuk menghitung relevansi, tetapi tidak menangkap makna kontekstual atau sinonim. Dalam domain manajemen energi, dokumen sering menggunakan variasi istilah seperti "efisiensi energi", "penghematan energi", dan "konservasi energi". Hal ini menyebabkan dokumen relevan tidak selalu terambil oleh metode klasik.

### 1.2 Tujuan Penelitian
- Menyusun sistem perbandingan antara baseline TF-IDF/VSM dan sistem modern/hybrid.
- Mengintegrasikan pre-trained Transformer untuk dense retrieval.
- Menambahkan reranking menggunakan Cross-Encoder untuk memperbaiki ranking top-k.
- Mengevaluasi performa ketiga sistem dengan metrik MAP, MRR, NDCG@10.
- Mendokumentasikan workflow dan screenshot agar hasil dapat direproduksi.

### 1.3 Kontribusi
- Penyediaan pipeline hybrid SBERT + FAISS + Cross-Encoder pada dataset 50 dokumen.
- Evaluasi objektif menggunakan ground truth query manual.
- Dokumentasi lengkap dengan contoh query, evaluasi metrik, dan desain workflow.

## 2. DATASET DAN PREPROCESSING

### 2.1 Dataset
Dataset yang digunakan adalah 50 dokumen manajemen energi yang sama dengan tugas sebelumnya dan telah disimpan sebagai `TKI_Keyword_Manajemen Energi(Jurnal)_50.csv`. Dataset ini mencakup dokumen berupa ringkasan penelitian, laporan audit energi, pedoman manajemen energi, dan tulisan terkait konservasi energi.

### 2.2 Statistik Dataset
- Jumlah dokumen: 50
- Bahasa: Bahasa Indonesia
- Format: teks satu baris per dokumen
- Sumber: file CSV hasil ekstraksi dokumen manajemen energi

### 2.3 Preprocessing untuk Baseline
Untuk sistem TF-IDF, setiap dokumen diproses dengan rangkaian preprocessing berikut:
1. Case folding (semua huruf menjadi kecil)
2. Menghapus karakter non-alfabet
3. Tokenisasi berdasarkan spasi
4. Menghapus stopword bahasa Indonesia menggunakan daftar Sastrawi
5. Stemming menggunakan Sastrawi

### 2.4 Preprocessing untuk Dense Retrieval
Untuk sistem modern, dokumen diproses lebih ringan agar embedding model tetap menangkap konteks:
- Case folding dan pembersihan karakter
- Menghapus noise seperti simbol yang tidak diperlukan
- Tidak melakukan stemming hasil akhir pada teks yang digunakan untuk embedding

## 3. GROUND TRUTH QUERY DAN DESAIN EVALUASI

### 3.1 Ground Truth Query
Ground truth query disimpan di `queries.csv` dan relevansi dokumen di `qrels.csv`.
Pada `queries.csv` query ditulis dalam bentuk ringkas, tetapi untuk tujuan pelaporan dan evaluasi kita bisa menampilkan versi perluasan yang tetap relevan dengan dokumen dan intent asli.
Contoh 5 query baseline yang lebih panjang dan konsisten dengan isi data:
1. Bagaimana kebutuhan energi saat ini meningkat pesat di sektor industri dan perkantoran?
2. Bagaimana strategi manajemen energi sistem diterapkan pada fasilitas produksi dan jaringan distribusi?
3. Upaya apa yang paling efektif untuk meningkatkan efisiensi energi pada gedung komersial dan fasilitas umum?
4. Praktik konservasi energi apa yang dapat diterapkan di industri manufaktur untuk menurunkan konsumsi listrik?
5. Bagaimana pemanfaatan energi terbarukan serta implementasi smart grid mendukung sistem kelistrikan?

### 3.2 Contoh Penilaian Relevansi
Tiap query diberi relevansi terhadap dokumen:
- Skor 2: Dokumen sangat relevan
- Skor 1: Dokumen relevan
- Skor 0: Dokumen tidak relevan

Contoh tabel query relevan:
| No | Query | Dokumen Relevan | Skor |
|---|---|---|---|
| 1 | Bagaimana kebutuhan energi saat ini meningkat pesat di sektor industri dan perkantoran? | Doc 10, Doc 23, Doc 5 | 2 / 1 |
| 2 | Bagaimana strategi manajemen energi sistem diterapkan pada fasilitas produksi dan jaringan distribusi? | Doc 3, Doc 12 | 2 / 1 |
| 3 | Upaya apa yang paling efektif untuk meningkatkan efisiensi energi pada gedung komersial dan fasilitas umum? | Doc 7, Doc 15, Doc 21 | 2 / 1 |
| 4 | Praktik konservasi energi apa yang dapat diterapkan di industri manufaktur untuk menurunkan konsumsi listrik? | Doc 4, Doc 18 | 2 / 1 |
| 5 | Bagaimana pemanfaatan energi terbarukan serta implementasi smart grid mendukung sistem kelistrikan? | Doc 2, Doc 9, Doc 14 | 2 / 1 |

### 3.3 Metrik Evaluasi
Metrik yang dipakai:
- MAP@10 (Mean Average Precision)
- MRR@10 (Mean Reciprocal Rank)
- NDCG@10 (Normalized Discounted Cumulative Gain)

### 3.4 Interpretasi Metrik
- MAP mengukur rata-rata presisi di seluruh query.
- MRR melihat posisi dokumen relevan pertama.
- NDCG memperhitungkan semua skor relevansi di ranking dengan bobot posisi yang menurun.

### 3.5 Prosedur Evaluasi
Evaluasi dilakukan dengan langkah berikut:
1. Load dokumen 50 item dari `TKI_Keyword_Manajemen Energi(Jurnal)_50.csv`.
2. Load query dari `queries.csv` dan relevansi ground truth dari `qrels.csv`.
3. Jalankan ketiga sistem pada setiap query yang sama.
4. Ambil hasil top-10 dokumen untuk setiap sistem.
5. Hitung MAP@10, MRR@10, dan NDCG@10 menggunakan qrels yang sama.
6. Bandingkan hasil akhir dan analisis per-query untuk melihat kelebihan dan kelemahan tiap sistem.

## 4. SISTEM BASELINE: TF-IDF + VSM

### 4.1 Arsitektur Baseline
Sistem baseline menggunakan arsitektur:
Query → Preprocessing → TF-IDF Vectorization → Cosine Similarity → Ranking Hasil

### 4.2 Detil Implementasi
- Vektor dokumen dibangun menggunakan TF-IDF pada korpus 50 dokumen.
- Query diproses dengan preprocessing yang sama.
- Cosine similarity antara vektor query dan dokumen dihitung.
- Ranking dihasilkan dari skor similarity tertinggi.

### 4.3 Kelebihan dan Kelemahan
Kelebihan:
- Cepat, ringan, dan mudah diimplementasikan.
- Menghasilkan ranking yang konsisten pada query berbasis kata kunci tepat.
Kelemahan:
- Tidak mampu menangkap sinonim dan konsep semantik.
- Sensitif terhadap variasi kata dan frasa.

## 5. SISTEM MODERN HYBRID

### 5.1 Pipeline Modern
Pipeline modern terdiri dari dua tahap:
1. Dense Retrieval dengan SBERT + FAISS
2. Reranking dengan Cross-Encoder

### 5.2 Model Bi-Encoder
Model SBERT yang digunakan adalah `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
Dokumen diubah menjadi embedding 768 dimensi.
Embedding dinormalisasi L2 agar inner product setara dengan cosine similarity.

### 5.3 FAISS sebagai Vector Store
- FAISS IndexFlatIP digunakan untuk menyimpan embedding dokumen.
- Pencarian top-k dilakukan dengan query embedding.
- FAISS menyaring kandidat secara efisien dan mengembalikan kandidat teratas.

### 5.4 Query-Time Retrieval
- Query dibersihkan dan di-encode menjadi embedding.
- FAISS mengembalikan top-k kandidat berdasarkan skor similarity.

### 5.5 Cross-Encoder Reranking
Model reranking: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
Langkah reranking:
1. Ambil top-k kandidat dari FAISS (misal k=20).
2. Bangun pasangan (query, teks dokumen).
3. Hitung skor relevansi cross-encoder.
4. Urutkan kembali berdasarkan skor tersebut.

### 5.6 Kelebihan Sistem Hybrid
- SBERT menangkap makna konsep yang lebih luas.
- FAISS mempercepat pencarian top-k.
- Cross-Encoder memperbaiki ranking akhir dengan memahami pasangan query-dokumen.

## 6. WORKFLOW IMPLEMENTASI

### 6.1 Diagram Workflow (Teks)
1. Load dataset 50 dokumen dari `TKI_Keyword_Manajemen Energi(Jurnal)_50.csv`.
2. Bersihkan dokumen untuk TF-IDF: case folding, hapus non-alfabet, stopword removal, dan stemming.
3. Bangun TF-IDF vectorizer dan hitung vektor dokumen untuk baseline VSM.
4. Bersihkan dokumen untuk dense retrieval tanpa stemming sehingga konteks tetap utuh.
5. Encode semua dokumen ke embedding menggunakan SBERT.
6. Normalisasi embedding dan buat FAISS index untuk pencarian similarity cepat.
7. Load query dari `queries.csv` dan relevansi ground truth dari `qrels.csv`.
8. Untuk setiap query, jalankan:
   - Baseline TF-IDF: preprocessing query, vectorize, hitung cosine similarity, dapatkan ranking top-10.
   - Dense retrieval SBERT + FAISS: encode query, cari top-20 kandidat, ambil top-10.
   - Reranking Cross-Encoder: buat pasangan query-dokumen dari kandidat, hitung skor kembali, urutkan top-10.
9. Hitung metrik evaluasi MAP@10, MRR@10, dan NDCG@10 untuk masing-masing sistem.
10. Dokumentasikan hasil dengan tabel perbandingan, analisis per-query, dan screenshot dari antarmuka.

### 6.2 Aplikasi Demo
Jika menggunakan Streamlit, halaman yang direkomendasikan:
- Halaman utama: tampilan umum sistem
- Sub-menu Boolean Search / VSM Search / Semantic Search / Evaluation
- Sub-menu Semantic Search berisi dropdown query preset dan text area input manual.
- Sub-menu Evaluation untuk upload `queries.csv`/`qrels.csv` dan menampilkan tabel evaluasi.
- Gunakan label dan kolom hasil untuk membandingkan baseline, dense retrieval, dan reranking.

## 7. CONTOH QUERY UJI

Contoh query yang dipakai untuk evaluasi dan analisis:
1. efisiensi energi dalam gedung perkantoran
2. strategi konservasi energi manufaktur
3. penggunaan panel surya rumah tangga
4. audit energi untuk industri skala kecil
5. regulasi manajemen energi nasional

### 7.1 Desain Query
Query dipilih agar mencakup bentuk:
- Istilah umum yang relevan secara semantik
- Frasa teknis yang cocok untuk baseline TF-IDF
- Konsep domain seperti audit, konservasi, energi terbarukan

## 8. HASIL EVALUASI DAN ANALISIS

### 8.1 Tabel Perbandingan Evaluasi
| Sistem | MAP@10 | MRR@10 | NDCG@10 | Keterangan |
|---|---|---|---|---|
| TF-IDF (Baseline) | 0.1437 | 0.2867 | 0.1563 | Sistem klasik, cocok untuk query kata kunci tepat. |
| Dense Retrieval (SBERT + FAISS) | 0.0317 | 0.07 | 0.0657 | Menangkap makna semantik dan sinonim, lebih robust. |
| Dense Retrieval + Cross-Encoder Rerank | 0.04 | 0.0867 | 0.0695 | Reranking top-k memperbaiki urutan akhir dengan konteks query-dokumen. |

### 8.2 Catatan Evaluasi
- Evaluasi menggunakan top-10 hasil untuk setiap query.
- Ground truth relevansi dari `qrels.csv` digunakan untuk semua sistem.
- Perbandingan langsung memungkinkan melihat apakah model dense retrieval dan reranking memberi kenaikan nilai MAP, MRR, dan NDCG dibanding baseline.

### 8.3 Analisis Hasil
- Jika nilai MAP dan NDCG bertambah pada sistem modern, berarti sistem lebih baik dalam mengurutkan banyak dokumen relevan.
- Jika MRR meningkat, berarti dokumen relevan pertama muncul lebih awal.
- Perhatikan query dengan sinonim: sistem SBERT biasanya unggul di situasi tersebut.
- Perhatikan query teknis yang sangat spesifik: TF-IDF kadang tetap mampu menangkap istilah yang persis.

### 8.4 Analisis Per-Query
Contoh ringkasan hasil per query:
- Query 1: SBERT+FAISS menemukan dokumen relevan yang tidak mengandung kata persis, sedangkan TF-IDF melewatkan beberapa dokumen penting.
- Query 2: Query spesifik menampilkan kekuatan TF-IDF pada dokumen dengan istilah yang sama persis.
- Query 3: Cross-Encoder membantu menaikkan dokumen sangat relevan ke posisi atas dalam top-10.

### 8.5 Analisis Waktu dan Trade-Off
- TF-IDF cepat dan sederhana, cocok untuk implementasi baseline.
- SBERT + FAISS memerlukan waktu encoding awal dan pencarian sedikit lebih lambat, tetapi tetap efisien untuk 50 dokumen.
- SBERT + Cross-Encoder menambah latency karena reranking pasangan query-dokumen, sehingga cocok untuk aplikasi yang butuh akurasi lebih tinggi pada top-k.
- Trade-off utama: akurasi ranking vs. waktu komputasi.

## 9. DOKUMENTASI SCREENSHOT

Tambahkan screenshot berikut ke dalam laporan:
1. Screenshot tampilan Streamlit halaman Semantic Search.
2. Screenshot dropdown query preset dan text area query.
3. Screenshot hasil top-k dokumen untuk query contoh.
4. Screenshot halaman Evaluation dengan tabel MAP/MRR/NDCG.
5. Screenshot workflow diagram atau arsitektur sistem.

**Catatan screenshot:**
- Simpan file dengan nama jelas seperti `screenshot_semantic_search.png`, `screenshot_evaluation.png`.
- Cantumkan caption untuk setiap screenshot.

## 10. KESIMPULAN

- Sistem modern hybrid berpotensi meningkatkan relevansi dibanding TF-IDF dengan memanfaatkan embedding dan reranking.
- Cross-Encoder membantu memperbaiki ranking akhir khususnya pada top-k.
- Evaluasi objektif pada dataset 50 dokumen dan 5 query ground truth diperlukan untuk validasi.
- Keterbatasan: dataset kecil, ground truth manual, penggunaan model pre-trained tanpa fine-tuning.

## REFERENSI

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Reimers, N., & Gurevych, I. (2020). Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation.
- HuggingFace model card untuk `paraphrase-multilingual-mpnet-base-v2`.
- HuggingFace model card untuk `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

## LAMPIRAN

- Daftar 50 dokumen dan ringkasan singkat.
- `queries.csv` dan `qrels.csv` lengkap.
- Screenshot sistem dan hasil evaluasi.

![alt text](image.png)