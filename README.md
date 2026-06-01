# Information Retrieval - Manajemen Energi

Aplikasi Information Retrieval menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)** dan **VSM (Vector Space Model)** untuk pencarian dokumen berbasis *keyword* "Manajemen Energi" dari data jurnal.

## Daftar Isi

1. [Deskripsi Project](https://www.google.com/search?q=%23deskripsi-project)
2. [Fitur Utama](https://www.google.com/search?q=%23fitur-utama)
3. [Struktur Project](https://www.google.com/search?q=%23struktur-project)
4. [Requirements](https://www.google.com/search?q=%23requirements)
5. [Instalasi](https://www.google.com/search?q=%23instalasi)
6. [Cara Menggunakan](https://www.google.com/search?q=%23cara-menggunakan)
7. [Alur Kerja (Pipeline)](https://www.google.com/search?q=%23alur-kerja-pipeline)
8. [Penjelasan Teknis](https://www.google.com/search?q=%23penjelasan-teknis)
9. [Output Data](https://www.google.com/search?q=%23output-data)
10. [Testing & Debugging](https://www.google.com/search?q=%23testing--debugging)
11. [Tips & Tricks](https://www.google.com/search?q=%23tips--tricks)
12. [Troubleshooting](https://www.google.com/search?q=%23troubleshooting)
13. [Referensi Teori](https://www.google.com/search?q=%23referensi-teori)

---

## Deskripsi Project

Proyek ini adalah implementasi sistem **Information Retrieval** yang menggabungkan teknik-teknik NLP (Natural Language Processing) dan *machine learning* dasar untuk:

* Memproses dokumen teks (berfokus pada jurnal tentang Manajemen Energi).
* Menghitung bobot TF-IDF untuk setiap term (kata).
* Melakukan pencarian dokumen berdasarkan *query* pengguna menggunakan *cosine similarity*.
* Menampilkan hasil pencarian yang relevan secara interaktif melalui antarmuka (UI) Streamlit.

---

## Fitur Utama

* **Preprocessing Teks Otomatis:**
* *Case folding* (mengubah teks menjadi huruf kecil).
* *Tokenisasi* (memecah kalimat menjadi kata).
* *Stopword removal* (menghapus kata-kata umum yang tidak memiliki makna pencarian spesifik).
* *Stemming* (mengubah kata ke bentuk dasar) menggunakan *library* Sastrawi.


* **Perhitungan TF-IDF:**
* Menghitung *Term Frequency* (TF) dengan *log frequency weighting*.
* Menghitung *Inverse Document Frequency* (IDF).
* Menghasilkan bobot TF-IDF untuk setiap term dalam dokumen.


* **Vector Space Model (VSM):**
* Representasi dokumen sebagai vektor.
* Perhitungan *cosine similarity*.
* Pemeringkatan (ranking) dokumen berdasarkan tingkat relevansi.


* **Antarmuka Interaktif:**
* Dibangun dengan Streamlit.
* Visualisasi hasil pencarian secara langsung.
* Menampilkan detail proses *preprocessing*.



---

## Struktur Project

```text
TKI/
├── README.md                                         # Dokumentasi project
├── app.py                                            # Aplikasi utama Streamlit
├── tfidf_vsm.py                                      # Modul TF-IDF dan VSM
├── preprocessing_manajemen_energi_&_tf_idf_vsm.ipynb # Notebook Jupyter (dokumentasi)
├── test_tfidf_vsm_colab.py                           # Script testing
├── test_compare.py                                   # Script perbandingan hasil
├── debug_csv.py                                      # Script debugging CSV
├── debug_tfidf_vsm.py                                # Script debugging TF-IDF
│
├── TKI_Keyword_Manajemen Energi(Jurnal).csv          # Data input (dokumen jurnal)
├── IR_ManajemenEnergi_Pipeline(Preprocessing & Index).csv
├── IR_KeywordManajemenEnergi(TF-IDF Scores).csv
└── (output files lainnya)

```

### Penjelasan File

| File | Deskripsi |
| --- | --- |
| `app.py` | Aplikasi utama Streamlit - *entry point* untuk menjalankan antarmuka |
| `tfidf_vsm.py` | Modul inti yang berisi fungsi-fungsi perhitungan TF-IDF dan VSM |
| `preprocessing_manajemen_energi_&_tf_idf_vsm.ipynb` | Dokumentasi berbentuk Jupyter Notebook yang menjelaskan *step-by-step* proses komputasi |
| `TKI_Keyword_Manajemen Energi(Jurnal).csv` | Dataset mentah berisi dokumen-dokumen jurnal tentang Manajemen Energi |

---

## Requirements

Pastikan Anda memiliki **Python 3.7+** dan telah menginstal *dependencies* berikut:

```text
streamlit>=1.28.0
pandas>=1.3.0
numpy>=1.21.0
Sastrawi>=1.0.1

```

---

## Instalasi

### Step 1: Clone/Copy Project ke Local Machine

```bash
# Pastikan Anda berada di dalam folder project
cd "l:\Semester 6\TKI"

```

### Step 2: Setup Python Environment (Disarankan)

**Windows (Command Prompt atau PowerShell):**

```bash
# Buat virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

```

**macOS/Linux:**

```bash
# Buat virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

```

### Step 3: Install Dependencies

```bash
# Menggunakan file requirements
pip install -r requirements.txt

```

Atau instal secara manual:

```bash
pip install streamlit pandas numpy Sastrawi

```

### Step 4: Verifikasi Instalasi

```bash
# Test ketersediaan modul utama
python -c "import streamlit; import pandas; import numpy; from Sastrawi.Stemmer.StemmerFactory import StemmerFactory; print('✓ Semua dependencies terinstall dengan baik')"

```

---

## Cara Menggunakan

### Menjalankan Aplikasi

```bash
# Pastikan virtual environment sudah aktif (jika menggunakan venv)
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run aplikasi
streamlit run app.py

```

Aplikasi akan otomatis terbuka di *browser* *default* Anda (biasanya di `http://localhost:8501`).

### Alur Penggunaan

1. **Input Query:** Ketik kata kunci yang ingin dicari (contoh: "energi terbarukan", "efisiensi energi", "manajemen konsumsi").
2. **Pantau Preprocessing:** Aplikasi akan menampilkan tahapan perubahan teks *query* mulai dari *case folding*, tokenisasi, *stopword removal*, hingga *stemming*.
3. **Evaluasi Hasil:** Dokumen akan diurutkan berdasarkan *relevance score*. Semakin tinggi angkanya, semakin relevan dokumen tersebut dengan *query* Anda.
4. **Ekspor Data (Opsional):** Hasil *preprocessing* dan nilai TF-IDF dapat diunduh (download) ke dalam format CSV melalui antarmuka.

---

## Alur Kerja (Pipeline)

### 1. Data Loading & Preprocessing

```text
CSV Input (TKI_Keyword_Manajemen Energi.csv)
    ↓
Case Folding (convert to lowercase, remove special chars)
    ↓
Tokenisasi (split text into words)
    ↓
Stopword Removal (remove common words)
    ↓
Stemming (convert words to base form using Sastrawi)
    ↓
Processed Documents (siap untuk TF-IDF calculation)

```

**Contoh Proses:**

* **Input:** "Sistem manajemen energi terbarukan yang efisien"
* **Case Folding:** "sistem manajemen energi terbarukan yang efisien"
* **Tokenisasi:** `["sistem", "manajemen", "energi", "terbarukan", "yang", "efisien"]`
* **Stopword Removal:** `["sistem", "manajemen", "energi", "terbarukan", "efisien"]` *(kata "yang" dihilangkan)*
* **Stemming:** `["sistem", "kelola", "energi", "terbarukan", "efisien"]` *(kata "manajemen" menjadi "kelola" tergantung kamus stemmer)*

### 2. Inverted Index Creation

Sistem membuat pemetaan dari setiap *term* menuju dokumen-dokumen yang memuatnya:

```json
{
  "energi": ["Doc 1", "Doc 3", "Doc 5"],
  "kelola": ["Doc 2", "Doc 4"],
  "sistem": ["Doc 1", "Doc 2", "Doc 3"]
}

```

### 3. Perhitungan TF-IDF

**TF (Term Frequency) - Log Weighting:**


$$TF(t,d) = 1 + \log_{10}(\text{raw\_tf})$$


*Dimana $\text{raw\_tf}$ adalah jumlah kemunculan term $t$ dalam dokumen $d$.*

**IDF (Inverse Document Frequency):**


$$IDF(t) = \log_{10}\left(\frac{N}{df_t}\right)$$


*Dimana $N$ adalah Total Dokumen dan $df_t$ adalah jumlah dokumen yang mengandung term $t$.*

**TF-IDF Akhir:**


$$TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)$$

### 4. Query Processing & Ranking

Pencarian dilakukan dengan menghitung jarak kedekatan menggunakan vektor cosinus:

$$\text{similarity}(q,d) = \frac{q \cdot d}{\|q\| \times \|d\|}$$

* $q \cdot d$ = *dot product* (hasil kali titik) antara vektor *query* dan dokumen.
* $\|q\|$ = magnitudo (norma) dari vektor *query*.
* $\|d\|$ = magnitudo (norma) dari vektor dokumen.

---

## Penjelasan Teknis

### Class & Functions di `tfidf_vsm.py`

#### 1. **TFIDFCalculator Class**

```python
class TFIDFCalculator:
    """
    Menghitung bobot TF-IDF untuk koleksi dokumen.
    """
    def __init__(self, records):
        # Inisialisasi dengan daftar dokumen yang telah di-preprocess
        pass
        
    def build_inverted_index(self):
        # Membangun dictionary pemetaan kata ke dokumen
        pass
        
    def calculate_idf(self):
        # Menghitung bobot IDF secara global
        pass
        
    def get_all_terms(self):
        # Mengembalikan list unik dari seluruh term dalam korpus
        pass

```

#### 2. **VSMCalculator Class**

```python
class VSMCalculator:
    """
    Mengeksekusi Vector Space Model untuk perankingan dokumen.
    """
    def query(self, query_text, tf_idf_doc_scores, idf_scores):
        # Mengonversi query, menghitung similarity, dan mereturn ranked list
        pass

```

### Preprocessing Pipeline di `app.py`

```python
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. Case Folding & Punctuation Removal
text = text.lower()
text = re.sub(r'[^a-z\s]', ' ', text)

# 2. Tokenisasi
tokens = text.split()

# 3. Stopword Removal
tokens = [t for t in tokens if t not in stopwords]

# 4. Stemming
stemmer = StemmerFactory().create_stemmer()
stemmed = [stemmer.stem(t) for t in tokens]

```

---

## Output Data

Aplikasi menghasilkan dua representasi data utama untuk analisis lanjutan:

### 1. Hasil Preprocessing

* **File:** `IR_ManajemenEnergi_Pipeline(Preprocessing & Index).csv`
* **Kolom Utama:** DocID, Teks Mentah, Hasil Case Folding, Tokenisasi, Filtering (Stopwords), dan Stemming.

### 2. Matriks Skor TF-IDF

* **File:** `IR_KeywordManajemenEnergi(TF-IDF Scores).csv`
* **Kolom Utama:** DocID, Term, Nilai TF, Nilai IDF, dan Total Skor TF-IDF.

---

## Testing & Debugging

Gunakan modul *script* bawaan berikut untuk mendiagnosis logika maupun dataset:

* `python test_tfidf_vsm_colab.py` : Verifikasi akurasi matematis TF-IDF dan kalkulasi VSM.
* `python test_compare.py` : Komparasi *output* aktual versus ekspektasi teoretis.
* `python debug_csv.py` : Melakukan *sanity check* pada pembacaan/penulisan file CSV.
* `python debug_tfidf_vsm.py` : Melacak kelainan (*anomaly*) pada perhitungan bobot spesifik per kata.

---

## Tips & Tricks

### Optimasi Query Pencarian

1. **Gunakan Kata Kunci Spesifik:** "sistem manajemen energi terbarukan" akan jauh lebih presisi dibandingkan sekadar "energi".
2. **Abaikan Stopwords Secara Manual:** Walaupun sistem menghapusnya secara otomatis, fokuskan pengetikan Anda pada inti topik.
3. **Kata Dasar:** Sistem Sastrawi akan mengonversi bentuk imbuhan (misal: "mengelola", "pengelolaan") menjadi kata dasarnya ("kelola"). Pemahaman ini membantu memprediksi kemunculan *term*.

### Performance Optimization

* Aplikasi memanfaatkan `@st.cache_data` pada Streamlit. Proses *preprocessing* dataset hanya terjadi di awal.
* Jika Anda memperbarui file CSV, pastikan untuk menghapus (*clear*) *cache* Streamlit (lewat ikon menu di kanan atas antarmuka UI) agar data ter-muat ulang.

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'Sastrawi'`

**Solusi:** Buka terminal/cmd dan jalankan `pip install Sastrawi`.

### Error: `FileNotFoundError: [Errno 2] No such file or directory: 'TKI_Keyword_Manajemen Energi(Jurnal).csv'`

**Solusi:**

* Pastikan file data mentah berada di *root folder* yang sama dengan `app.py`.
* Periksa kembali nama file (ingat bahwa penamaan file bersifat *case-sensitive* terutama pada sistem Linux/macOS).

### Aplikasi Streamlit Gagal Terbuka (Port Tabrakan)

**Solusi:** Paksa Streamlit menggunakan *port* lain yang kosong.

```bash
streamlit run app.py --server.port 8502

```

### Hasil Pencarian Selalu Kosong

**Solusi:**

* Bisa jadi *query* Anda murni hanya terdiri dari *stopwords* (contoh: "dan", "yang", "di").
* Terjadi *over-stemming* sehingga kata dasar yang dihasilkan tidak *match* dengan dokumen. Coba kata kunci alternatif yang lebih universal.

---

## Referensi Teori

* **TF-IDF (Term Frequency - Inverse Document Frequency):** Algoritma pembobotan untuk mengevaluasi seberapa esensial sebuah kata (term) terhadap sebuah dokumen di dalam kumpulan korpus (dataset).
* **VSM (Vector Space Model):** Model aljabar yang merepresentasikan dokumen dan *query* teks sebagai vektor dari berbagai indikator identitas (seperti bobot TF-IDF).
* **Cosine Similarity:** Metrik jarak geometris di dalam ruang multidimensi vektor yang dinilai dari sudut ($\theta$) antara dua vektor. Rentang nilainya antara $0$ (sangat bertolak belakang/tidak berhubungan) hingga $1$ (identik secara proporsional).

$$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \times \|\mathbf{B}\|}$$

---
