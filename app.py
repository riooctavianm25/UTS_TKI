import streamlit as st
import pandas as pd
import re
import math
from collections import defaultdict
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(page_title="Information Retrieval - Manajemen Energi", layout="wide")

@st.cache_resource
def initialize_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_data
def load_stopwords():
    return {
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

@st.cache_data
def load_data():
    """Load and preprocess data"""
    stemmer = initialize_stemmer()
    stopwords = load_stopwords()
    
    # Load sentences
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

    return records, sentences, stemmer, stopwords

st.title("Information Retrieval Dashboard")
st.markdown("### Manajemen Energi - Text Preprocessing & Search")

with st.spinner("Loading data..."):
    records, sentences, stemmer, stopwords = load_data()

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Documents", len(records))
with col2:
    st.metric("Total Sentences", len(sentences))
with col3:
    total_tokens = sum(len(rec['Stemming']) for rec in records)
    st.metric("Total Tokens (after stemming)", total_tokens)
with col4:
    unique_terms = len(set(t for rec in records for t in rec['Stemming']))
    st.metric("Unique Terms", unique_terms)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Preprocessing", "🔍 Inverted Index", "🔄 Forward Index", "⚙️ Boolean Search", "🎯 VSM Search"]
)

# TAB 1: PREPROCESSING
with tab1:
    st.subheader("Text Preprocessing Pipeline")
    
    # Display table with key columns
    df_display = pd.DataFrame([{
        'DocID': rec['DocID'],
        'Teks Mentah': rec['Teks Mentah'][:60] + '...',
        'Tokens': ', '.join(rec['Tokenisasi'][:5]) + ('...' if len(rec['Tokenisasi']) > 5 else ''),
        'After Stemming': ', '.join(rec['Stemming'][:5]) + ('...' if len(rec['Stemming']) > 5 else ''),
        'Unique Terms': len(rec['Stemming'])
    } for rec in records])
    
    st.dataframe(df_display, use_container_width=True)
    
    # Detail view
    st.markdown("---")
    st.subheader("Detail View")
    doc_select = st.selectbox("Pilih dokumen:", [rec['DocID'] for rec in records])
    selected_rec = next(r for r in records if r['DocID'] == doc_select)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Original Text**")
        st.write(selected_rec['Teks Mentah'])
    with col2:
        st.info("**After Processing**")
        st.write(f"**Stemmed Terms:** {', '.join(selected_rec['Stemming'])}")

# TAB 2: INVERTED INDEX
with tab2:
    st.subheader("Inverted Index")
    
    # Build inverted index
    inverted_index = defaultdict(list)
    for rec in records:
        for term in set(rec['Stemming']):
            inverted_index[rec['DocID']].append(term) if rec['DocID'] not in inverted_index else None
    
    inverted_sorted = {}
    for rec in records:
        for term in set(rec['Stemming']):
            if term not in inverted_sorted:
                inverted_sorted[term] = []
            inverted_sorted[term].append(rec['DocID'])
    
    # Display
    ii_data = []
    for i, (term, posting) in enumerate(sorted(inverted_sorted.items())[:30], 1):
        ii_data.append({
            'No': i,
            'Kata Dasar': term,
            'Posting List': ', '.join(posting),
            'DF': len(posting)
        })
    
    df_ii = pd.DataFrame(ii_data)
    st.dataframe(df_ii, use_container_width=True)
    st.caption(f"Total unique terms: {len(inverted_sorted)} (showing first 30)")

# TAB 3: FORWARD INDEX
with tab3:
    st.subheader("Forward Index")
    
    fi_data = []
    for rec in records:
        freq = defaultdict(int)
        for term in rec['Stemming']:
            freq[term] += 1
        
        top_terms = sorted(freq.items(), key=lambda x: -x[1])[:5]
        term_freq_str = ', '.join([f'{t}:{f}' for t, f in top_terms])
        
        fi_data.append({
            'DocID': rec['DocID'],
            'Top Terms (Freq)': term_freq_str,
            'Total Unique': len(freq)
        })
    
    df_fi = pd.DataFrame(fi_data)
    st.dataframe(df_fi, use_container_width=True)

# TAB 4: BOOLEAN SEARCH
with tab4:
    st.subheader("Boolean Query Search")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query_type = st.radio("Query Type:", ["AND Query", "OR Query"])
    
    if query_type == "AND Query":
        terms = st.multiselect("Select terms (semua harus ada):", 
                              sorted(set(t for rec in records for t in rec['Stemming']))[:50])
        
        if terms:
            # AND operation
            results = set(rec['DocID'] for rec in records 
                         if all(stemmer.stem(t.lower()) in rec['Stemming'] for t in terms))
            
            st.write(f"**Query:** {' AND '.join(terms)}")
            if results:
                st.success(f"Ditemukan {len(results)} dokumen")
                st.write(", ".join(sorted(results)))
            else:
                st.warning("Tidak ada dokumen yang cocok")
    
    else:  # OR Query
        terms = st.multiselect("Select terms (salah satu boleh ada):", 
                              sorted(set(t for rec in records for t in rec['Stemming']))[:50])
        
        if terms:
            # OR operation
            results = set(rec['DocID'] for rec in records 
                         if any(stemmer.stem(t.lower()) in rec['Stemming'] for t in terms))
            
            st.write(f"**Query:** {' OR '.join(terms)}")
            if results:
                st.success(f"Ditemukan {len(results)} dokumen")
                st.write(", ".join(sorted(results)))
            else:
                st.warning("Tidak ada dokumen yang cocok")

# TAB 5: VSM SEARCH
with tab5:
    st.subheader("Vector Space Model Search")
    
    query = st.text_input("Masukkan query Anda:")
    
    if query:
        # Process query
        query_rec = {
            'DocID': 'Query',
            'Teks Mentah': query,
            'Case Folding': '',
            'Tokenisasi': [],
            'Stopword Removal': [],
            'Stemming': []
        }
        
        # Preprocessing query
        folded = re.sub(r'[^a-z\s]', ' ', query.lower())
        folded = re.sub(r'\s+', ' ', folded).strip()
        tokens = [t for t in folded.split() if len(t) >= 3]
        no_stop = [t for t in tokens if t not in stopwords]
        
        stemmed = []
        for t in no_stop:
            stem = stemmer.stem(t)
            if stem not in stopwords and len(stem) >= 3:
                stemmed.append(stem)
        
        query_rec['Stemming'] = stemmed
        
        # Calculate TF-IDF similarity
        if stemmed:
            similarities = []
            
            for rec in records:
                # Simple Jaccard similarity
                query_terms = set(stemmed)
                doc_terms = set(rec['Stemming'])
                
                if query_terms and doc_terms:
                    intersection = len(query_terms & doc_terms)
                    union = len(query_terms | doc_terms)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0:
                        similarities.append({
                            'DocID': rec['DocID'],
                            'Similarity': f"{similarity:.3f}",
                            'Preview': rec['Teks Mentah'][:80] + '...'
                        })
            
            if similarities:
                df_sim = pd.DataFrame(similarities).sort_values('Similarity', ascending=False)
                st.dataframe(df_sim, use_container_width=True)
                st.success(f"✓ Found {len(df_sim)} relevant documents")
            else:
                st.warning("No relevant documents found")
        else:
            st.info("Query terlalu umum (stopwords removed)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Information Retrieval System | Manajemen Energi Dataset | Built with Streamlit
</div>
""", unsafe_allow_html=True)