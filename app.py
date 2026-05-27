import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import defaultdict
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tfidf_vsm import TFIDFCalculator, VSMCalculator

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

def calculate_tfidf(records):
    """Calculate TF-IDF matrix using the new TFIDFCalculator"""
    tfidf_calc = TFIDFCalculator(records)
    tfidf_calc.build_inverted_index()
    tfidf_calc.calculate_idf()
    tfidf_calc.calculate_tf_idf()
    
    # Convert to format compatible with existing code
    tfidf_matrix = []
    for rec in records:
        doc_id = rec['DocID']
        tfidf_matrix.append(tfidf_calc.tf_idf_doc_scores.get(doc_id, {}))
    
    return tfidf_matrix, tfidf_calc.idf_scores, tfidf_calc.all_terms, tfidf_calc

def style_dataframe_gradient(df, columns=None):
    """Apply background gradient to dataframe"""
    def gradient_color(val):
        if isinstance(val, str):
            return ''
        if val == 0:
            return 'background-color: #ffe6f0'
        normalized = min(val / (df[df.columns[0]].max()) if df[df.columns[0]].max() > 0 else 1, 1)
        return f'background-color: rgba(144, 238, 144, {normalized})'
    
    if columns is None:
        columns = df.columns
    return df.style.applymap(lambda x: gradient_color(x) if isinstance(x, (int, float)) else '')

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
main_menu = st.sidebar.radio("Main Menu", ["📄 Shelf Monitoring", "⚙️ Search Engine"])

if main_menu == "📄 Shelf Monitoring":
    sub_menu = st.sidebar.radio("Sub-Menu", ["Preprocessing", "Inverted Index", "Forward Index"])
else:
    sub_menu = st.sidebar.radio("Sub-Menu", ["Boolean Search", "VSM Search"])

# Load data
with st.spinner("Loading data..."):
    records, sentences, stemmer, stopwords = load_data()
    tfidf_matrix, idf_dict, all_terms, tfidf_calc = calculate_tfidf(records)

# Top Section: Search & Filters
st.title("📊 Information Retrieval System")
st.markdown("### Manajemen Energi Dataset")

# Search box and filters
col_search, col_filter1, col_filter2 = st.columns([2, 1, 1])
with col_search:
    search_query = st.text_input("", placeholder="Search...")
with col_filter1:
    retailer_filter = st.selectbox("", ["All Retailers"], key="retailer")
with col_filter2:
    category_filter = st.selectbox("", ["All Categories"], key="category")

st.markdown("---")

# KPI Cards
kpi_col1, kpi_col2 = st.columns(2)

total_docs = len(records)
unique_terms = len(all_terms)
keyword_density = unique_terms / total_docs if total_docs > 0 else 0

with kpi_col1:
    st.metric("📈 Total Documents Processed", f"{total_docs}", "Shelf Items")

with kpi_col2:
    st.metric("📚 Keyword Index Density", f"{keyword_density:.2f}", "Terms per Doc")

st.markdown("---")


# ============================================================================
# SHELF MONITORING SECTION
# ============================================================================

if main_menu == "📄 Shelf Monitoring":
    
    # ========== PREPROCESSING ==========
    if sub_menu == "Preprocessing":
        st.subheader("📋 Text Preprocessing Pipeline")
        st.markdown("Visualization of text preprocessing stages from raw text to stemming")
        
        # Display table with conditional formatting
        df_display = pd.DataFrame([{
            'DocID': rec['DocID'],
            'Teks Mentah': rec['Teks Mentah'][:60] + '...',
            'Tokens': len(rec['Tokenisasi']),
            'After Stemming': len(rec['Stemming']),
            'Unique Terms': len(set(rec['Stemming']))
        } for rec in records])
        
        # Apply gradient styling
        styled_df = df_display.style.background_gradient(
            subset=['Tokens', 'After Stemming', 'Unique Terms'],
            cmap='RdYlGn',
            vmin=0,
            vmax=max(df_display['Unique Terms'].max(), 1)
        ).format({
            'Tokens': '{:.0f}',
            'After Stemming': '{:.0f}',
            'Unique Terms': '{:.0f}'
        })
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.markdown("---")
        st.subheader("📖 Detail View")
        
        doc_select = st.selectbox("Select Document:", 
                                  options=[rec['DocID'] for rec in records],
                                  key="preprocessing_select")
        selected_rec = next(r for r in records if r['DocID'] == doc_select)
        
        tab_original, tab_processed, tab_pipeline = st.tabs(["Original", "Final Result", "Pipeline"])
        
        with tab_original:
            st.info("**Original Text**")
            st.write(selected_rec['Teks Mentah'])
        
        with tab_processed:
            st.success("**After Complete Processing (Stemming)**")
            st.write(f"**Stemmed Terms ({len(selected_rec['Stemming'])} terms):**")
            st.write(", ".join(selected_rec['Stemming']) if selected_rec['Stemming'] else "No terms")
        
        with tab_pipeline:
            st.markdown("**Processing Pipeline:**")
            cols = st.columns(5)
            with cols[0]:
                st.write("**1. Case Folding**")
                st.caption(selected_rec['Case Folding'][:100] + '...')
            with cols[1]:
                st.write("**2. Tokenization**")
                st.caption(f"{len(selected_rec['Tokenisasi'])} tokens")
            with cols[2]:
                st.write("**3. Stopword Removal**")
                st.caption(f"{len(selected_rec['Stopword Removal'])} terms")
            with cols[3]:
                st.write("**4. Stemming**")
                st.caption(f"{len(selected_rec['Stemming'])} terms")
            with cols[4]:
                st.write("**5. Final**")
                st.caption(f"Ready for indexing")
    
    # ========== INVERTED INDEX ==========
    elif sub_menu == "Inverted Index":
        st.subheader("🔍 Inverted Index Structure")
        st.markdown("Map of keywords to documents containing them")
        
        # Build inverted index
        inverted_index = defaultdict(list)
        term_freq = defaultdict(int)
        
        for rec in records:
            for term in set(rec['Stemming']):
                inverted_index[term].append(rec['DocID'])
                term_freq[term] += rec['Stemming'].count(term)
        
        # Display top terms
        ii_data = []
        for i, term in enumerate(sorted(inverted_index.keys())[:50], 1):
            posting = inverted_index[term]
            ii_data.append({
                'No': i,
                'Term': term,
                'Document Frequency': len(posting),
                'Term Frequency': term_freq[term],
                'Posting List': ', '.join(posting)
            })
        
        df_ii = pd.DataFrame(ii_data)
        
        # Apply gradient styling
        styled_ii = df_ii.style.background_gradient(
            subset=['Document Frequency', 'Term Frequency'],
            cmap='YlOrRd',
            vmin=0,
            vmax=max(df_ii['Document Frequency'].max(), df_ii['Term Frequency'].max(), 1)
        )
        
        st.dataframe(styled_ii, use_container_width=True, height=400)
        st.caption(f"Total unique terms: {len(inverted_index)} (showing first 50)")
        
        # Statistics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Terms", len(inverted_index))
        with col2:
            avg_df = sum(len(docs) for docs in inverted_index.values()) / len(inverted_index)
            st.metric("Avg Document Frequency", f"{avg_df:.2f}")
        with col3:
            max_df = max(len(docs) for docs in inverted_index.values())
            st.metric("Max Document Frequency", max_df)
        with col4:
            total_tf = sum(term_freq.values())
            st.metric("Total Term Frequency", total_tf)
    
    # ========== FORWARD INDEX ==========
    elif sub_menu == "Forward Index":
        st.subheader("📑 Forward Index Structure")
        st.markdown("Map of documents to their terms with frequencies")
        
        fi_data = []
        for rec in records:
            freq = defaultdict(int)
            for term in rec['Stemming']:
                freq[term] += 1
            
            top_terms = sorted(freq.items(), key=lambda x: -x[1])[:10]
            unique_count = len(freq)
            total_terms = len(rec['Stemming'])
            
            fi_data.append({
                'DocID': rec['DocID'],
                'Unique Terms': unique_count,
                'Total Terms': total_terms,
                'Top 5 Terms': ', '.join([f'{t}({f})' for t, f in top_terms[:5]]),
                'Density': unique_count / total_terms if total_terms > 0 else 0
            })
        
        df_fi = pd.DataFrame(fi_data)
        
        # Apply gradient styling
        styled_fi = df_fi.style.background_gradient(
            subset=['Unique Terms', 'Total Terms', 'Density'],
            cmap='Blues',
            vmin=0,
            vmax=max(df_fi['Unique Terms'].max(), 1)
        )
        
        st.dataframe(styled_fi, use_container_width=True, height=400)
        
        st.markdown("---")
        st.subheader("📊 Detailed Term Frequency Analysis")
        
        doc_select = st.selectbox("Select Document:", 
                                  options=[rec['DocID'] for rec in records],
                                  key="forward_select")
        selected_rec = next(r for r in records if r['DocID'] == doc_select)
        
        freq = defaultdict(int)
        for term in selected_rec['Stemming']:
            freq[term] += 1
        
        freq_data = sorted(freq.items(), key=lambda x: -x[1])
        
        if freq_data:
            df_freq = pd.DataFrame(freq_data, columns=['Term', 'Frequency'])
            
            styled_freq = df_freq.style.background_gradient(
                subset=['Frequency'],
                cmap='Greens',
                vmin=0,
                vmax=df_freq['Frequency'].max()
            )
            
            st.dataframe(styled_freq, use_container_width=True)
        else:
            st.info("No terms in this document")

# ============================================================================
# SEARCH ENGINE SECTION
# ============================================================================

else:
    
    # ========== BOOLEAN SEARCH ==========
    if sub_menu == "Boolean Search":
        st.subheader("🔎 Boolean Query Search")
        st.markdown("Search using AND/OR boolean operations")
        
        query_type = st.radio("Query Type:", 
                             options=["AND Query", "OR Query"],
                             horizontal=True)
        
        # Get all unique terms for selection
        all_unique_terms = sorted(set(t for rec in records for t in rec['Stemming']))
        
        if query_type == "AND Query":
            st.info("**AND Query:** Returns documents containing ALL selected terms")
            selected_terms = st.multiselect(
                "Select terms (all must be present):", 
                options=all_unique_terms,
                key="boolean_and"
            )
            
            if selected_terms:
                # AND operation
                results = []
                for rec in records:
                    rec_terms = set(rec['Stemming'])
                    if all(term in rec_terms for term in selected_terms):
                        results.append({
                            'DocID': rec['DocID'],
                            'Text Preview': rec['Teks Mentah'][:100] + '...',
                            'Matching Terms': ', '.join(t for t in selected_terms if t in rec_terms)
                        })
                
                st.markdown(f"**Query:** {' AND '.join([f'**{t}**' for t in selected_terms])}")
                
                if results:
                    st.success(f"✅ Found {len(results)} documents")
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                else:
                    st.warning("❌ No documents found matching all terms")
        
        else:  # OR Query
            st.info("**OR Query:** Returns documents containing ANY of the selected terms")
            selected_terms = st.multiselect(
                "Select terms (at least one must be present):", 
                options=all_unique_terms,
                key="boolean_or"
            )
            
            if selected_terms:
                # OR operation
                results = []
                for rec in records:
                    rec_terms = set(rec['Stemming'])
                    if any(term in rec_terms for term in selected_terms):
                        matching = [t for t in selected_terms if t in rec_terms]
                        results.append({
                            'DocID': rec['DocID'],
                            'Text Preview': rec['Teks Mentah'][:100] + '...',
                            'Matching Terms': ', '.join(matching)
                        })
                
                st.markdown(f"**Query:** {' OR '.join([f'**{t}**' for t in selected_terms])}")
                
                if results:
                    st.success(f"✅ Found {len(results)} documents")
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                else:
                    st.warning("❌ No documents found matching any term")
    
    # ========== VSM SEARCH ==========
    elif sub_menu == "VSM Search":
        st.subheader("🎯 Vector Space Model Search (TF-IDF + Cosine Similarity)")
        st.markdown("Search using TF-IDF weighting and Cosine Similarity")
        
        # Query input
        query_text = st.text_area(
            "Enter your query:",
            placeholder="e.g., manajemen energi sistem",
            height=80
        )
        
        search_threshold = st.slider(
            "Similarity Threshold (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05
        )
        
        if query_text:
            # Gunakan VSMCalculator dengan stemmer dan stopwords
            vsm_calc = VSMCalculator(tfidf_calc, stemmer, stopwords)
            results, query_terms = vsm_calc.get_detailed_results(
                query_text, 
                records,
                threshold=search_threshold
            )
            
            st.markdown(f"**Query Terms:** {', '.join([f'`{t}`' for t in query_terms])}")
            
            if results:
                st.success(f"✅ Found {len(results)} documents")
                
                # Display results table
                display_results = []
                for r in results:
                    display_results.append({
                        'DocID': r['DocID'],
                        'Similarity Score': f"{r['Similarity Score']:.4f}",
                        'Similarity (%)': f"{r['Similarity Score']*100:.2f}%",
                        'Text Preview': r['Text Preview']
                    })
                
                df_results = pd.DataFrame(display_results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Export detailed results
                with st.expander("📊 Detailed Results"):
                    for i, r in enumerate(results[:5], 1):
                        st.markdown(f"**{i}. {r['DocID']} - Similarity: {r['Similarity Score']:.4f}**")
                        st.text(r['Full Text'])
                        st.divider()
                
                # Visualization
                st.markdown("---")
                st.subheader("📈 Similarity Scores Chart")
                
                chart_data = []
                for r in results[:10]:
                    chart_data.append({
                        'Document': r['DocID'],
                        'Score': r['Similarity Score'] * 100
                    })
                
                df_chart = pd.DataFrame(chart_data)
                if not df_chart.empty:
                    st.bar_chart(df_chart.set_index('Document')['Score'])
            else:
                st.warning(f"❌ No documents found with similarity >= {search_threshold:.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Information Retrieval System | Manajemen Energi Dataset | TF-IDF + Cosine Similarity | Built with Streamlit
</div>
""", unsafe_allow_html=True)
