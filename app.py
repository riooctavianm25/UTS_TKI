import os
import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import defaultdict
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tfidf_vsm import TFIDFCalculator, VSMCalculator, SemanticSearch
from dense_retrieval import DenseRetrievalEngine

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


def clean_text_for_sbert(text):
    """Buat representasi teks natural untuk SBERT/BERT tanpa stemming atau stopword removal."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_tfidf(text, stemmer, stopwords):
    """Pipeline preprocessing untuk TF-IDF/VSM."""
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

    folded = case_folding(text)
    tokens = tokenisasi(folded)
    no_stop = hapus_stopword(tokens)
    stemmed = stemming(no_stop)
    return {
        'Case Folding': folded,
        'Tokenisasi': tokens,
        'Stopword Removal': no_stop,
        'Stemming': stemmed,
    }


def preprocess_sbert(text):
    """Pipeline preprocessing ringan untuk SBERT/BERT."""
    return clean_text_for_sbert(text)


@st.cache_data
def load_queries_file(path='queries.csv'):
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
        cols = [c.lower() for c in df.columns]
        if 'query_id' not in cols or 'query_text' not in cols:
            return {}
        queries = {}
        for _, row in df.iterrows():
            qid = str(row['query_id']).strip()
            qtext = str(row['query_text']).strip()
            if qid and qtext:
                queries[qid] = qtext
        return queries
    except Exception:
        return {}

@st.cache_data
def load_data():
    """Load and preprocess data"""
    stemmer = initialize_stemmer()
    stopwords = load_stopwords()
    
    # Prefer 50-document dataset if available
    data_path = 'TKI_Keyword_Manajemen Energi(Jurnal)_50.csv'
    if not os.path.exists(data_path):
        data_path = 'TKI_Keyword_Manajemen Energi(Jurnal).csv'

    df = pd.read_csv(data_path, encoding='utf-8-sig')
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

    def preprocessing(doc_id, raw_text):
        tfidf_result = preprocess_tfidf(raw_text, stemmer, stopwords)
        sbert_text = preprocess_sbert(raw_text)
        return {
            'DocID': doc_id,
            'Teks Mentah': raw_text.strip(),
            'Case Folding': tfidf_result['Case Folding'],
            'Tokenisasi': tfidf_result['Tokenisasi'],
            'Stopword Removal': tfidf_result['Stopword Removal'],
            'Stemming': tfidf_result['Stemming'],
            'SBERT Clean Text': sbert_text,
        }

    records = []
    for i, kalimat in enumerate(sentences[:50], 1):
        record = preprocessing(f'Doc {i}', kalimat)
        records.append(record)

    return records, sentences[:50], stemmer, stopwords

@st.cache_resource
def initialize_semantic_search(records):
    return SemanticSearch(records, use_rerank=False)


@st.cache_resource
def initialize_dense_engine(records):
    """Eagerly load SBERT bi-encoder, encode documents and build FAISS index.

    If any heavy dependency is missing this will show a warning but still
    return a partially initialized engine object to avoid crashing the app.
    """
    engine = DenseRetrievalEngine(records)
    try:
        engine.load_bi_encoder()
    except Exception as e:
        st.warning(f"Could not load SBERT bi-encoder: {e}")
        return engine

    try:
        engine.encode_documents()
        engine.build_faiss_index()
    except Exception as e:
        st.warning(f"Error while encoding documents or building FAISS index: {e}")

    try:
        # Optional: attempt to load a cross-encoder reranker if available
        engine.load_reranker()
        st.info("Cross-Encoder loaded for optional reranking.")
    except Exception:
        # Non-fatal: continue without a reranker
        pass

    return engine


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


def build_hybrid_candidate_ranking(tfidf_candidates, dense_hits, top_k, max_candidates=20):
    """Bangun ranking gabungan lexical-semantic agar evaluasi SBERT lebih adil."""
    if not tfidf_candidates and not dense_hits:
        return []

    lexical_ids = [doc_id for doc_id in tfidf_candidates[:max_candidates] if doc_id]
    semantic_ids = [getattr(hit, 'doc_id', None) for hit in dense_hits[:max_candidates] if getattr(hit, 'doc_id', None)]

    ordered_ids = []
    seen = set()
    for doc_id in lexical_ids + semantic_ids:
        if doc_id and doc_id not in seen:
            ordered_ids.append(doc_id)
            seen.add(doc_id)

    return ordered_ids[:top_k]

# Sidebar Navigation
st.sidebar.title("Navigation")
main_menu = st.sidebar.radio("Main Menu", ["Shelf Monitoring", "Search Engine"])

if main_menu == "Shelf Monitoring":
    sub_menu = st.sidebar.radio("Sub-Menu", ["Preprocessing", "Inverted Index", "Forward Index"])
else:
    sub_menu = st.sidebar.radio("Sub-Menu", ["Boolean Search", "VSM Search", "Semantic Search", "Evaluation"])

# Load data
with st.spinner("Loading data..."):
    records, sentences, stemmer, stopwords = load_data()
    tfidf_matrix, idf_dict, all_terms, tfidf_calc = calculate_tfidf(records)
    semantic_search = initialize_semantic_search(records)
    # Eagerly initialize dense retrieval (SBERT + FAISS) at startup
    dense_engine = initialize_dense_engine(records)

# Top Section: Search & Filters
st.title("Information Retrieval System")
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
    st.metric("Total Documents Processed", f"{total_docs}", "Shelf Items")

with kpi_col2:
    st.metric("Keyword Index Density", f"{keyword_density:.2f}", "Terms per Doc")

st.markdown("---")


# ============================================================================
# SHELF MONITORING SECTION
# ============================================================================

if main_menu == "Shelf Monitoring":
    
    # ========== PREPROCESSING ==========
    if sub_menu == "Preprocessing":
        st.subheader("Text Preprocessing Pipeline")
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
        st.subheader("Detail View")
        
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
        st.subheader("Inverted Index Structure")
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
        st.subheader("Forward Index Structure")
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
        st.subheader("Detailed Term Frequency Analysis")
        
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
        st.subheader("Boolean Query Search")
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
                    st.success(f"Found {len(results)} documents")
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                else:
                    st.warning("No documents found matching all terms")
        
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
                    st.success(f"Found {len(results)} documents")
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                else:
                    st.warning("No documents found matching any term")
    
    # ========== VSM SEARCH ==========
    elif sub_menu == "VSM Search":
        st.subheader("Vector Space Model Search (TF-IDF + Cosine Similarity)")
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
                st.success(f"Found {len(results)} documents")
                
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
                with st.expander("Detailed Results"):
                    for i, r in enumerate(results[:5], 1):
                        st.markdown(f"**{i}. {r['DocID']} - Similarity: {r['Similarity Score']:.4f}**")
                        st.text(r['Full Text'])
                        st.divider()
                
                # Visualization
                st.markdown("---")
                st.subheader("Similarity Scores Chart")
                
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
                st.warning(f"No documents found with similarity >= {search_threshold:.2f}")
    
    elif sub_menu == "Semantic Search":
        st.subheader("Semantic Search (SBERT + FAISS)")
        st.markdown("Search using dense document embeddings and approximate nearest neighbor retrieval.")

        query_file_queries = load_queries_file('queries.csv')
        query_text = ""

        if query_file_queries:
            manual_label = "-- Ketik manual --"
            query_options = [manual_label] + [f"{qid}: {qtext}" for qid, qtext in sorted(query_file_queries.items(), key=lambda item: item[0])]
            selected = st.selectbox("", query_options, index=0, label_visibility="collapsed")
            if selected != manual_label:
                query_text = selected.split(":", 1)[1].strip()
            st.caption("Pilih preset dari dropdown lalu edit teks di bawah. Jika ingin manual, pilih -- Ketik manual --.")

        query_text = st.text_area(
            "Enter your query:",
            value=query_text,
            placeholder="Pilih preset di atas atau ketik query Anda di sini...",
            height=80
        )

        top_k = st.slider(
            "Top K results:",
            min_value=1,
            max_value=20,
            value=10,
            step=1
        )

        enable_rerank = st.checkbox("Enable Cross-Encoder Reranking (if available)", value=False)

        if query_text:
            with st.spinner("Running semantic search..."):
                    # Use dense_engine (SBERT + FAISS) initialized at startup
                    try:
                        faiss_results = dense_engine.search(query_text, top_k=top_k)
                    except Exception as e:
                        st.error(f"Dense retrieval failed: {e}")
                        faiss_results = []

                    # Keep original FAISS scores for display
                    original_scores = {r.doc_id: r.score for r in faiss_results}

                    # Optionally rerank with Cross-Encoder if requested and available
                    if enable_rerank:
                        if getattr(dense_engine, 'reranker', None) is not None:
                            try:
                                results = dense_engine.rerank(query_text, faiss_results)
                            except Exception as e:
                                st.warning(f"Reranking failed: {e}")
                                results = faiss_results
                        else:
                            st.warning("Cross-Encoder not available; showing FAISS results.")
                            results = faiss_results
                    else:
                        results = faiss_results

            if results:
                st.success(f"Found {len(results)} documents")

                def map_cosine_to_01(c):
                    try:
                        return (c + 1.0) / 2.0
                    except Exception:
                        return 0.0

                def sigmoid(x):
                    try:
                        return 1.0 / (1.0 + math.exp(-x))
                    except OverflowError:
                        return 0.0 if x < 0 else 1.0

                display_results = []
                for r in results:
                    doc_id = getattr(r, 'doc_id', getattr(r, 'DocID', None))
                    text_preview = getattr(r, 'text_preview', getattr(r, 'Text Preview', ''))
                    score_raw = getattr(r, 'score', getattr(r, 'Score', 0.0))
                    # If this result was reranked, also show original FAISS cosine
                    orig = original_scores.get(doc_id)
                    if orig is not None:
                        cosine = float(orig)
                        pct = map_cosine_to_01(cosine) * 100.0
                    else:
                        cosine = float(score_raw)
                        pct = map_cosine_to_01(cosine) * 100.0

                    row = {
                        'DocID': doc_id,
                        'Cosine': f"{cosine:.4f}",
                        'Similarity (%)': f"{pct:.2f}%",
                        'Text Preview': text_preview,
                    }
                    # If reranked, add cross-encoder score column
                    if enable_rerank and getattr(dense_engine, 'reranker', None) is not None:
                        rerank_raw = float(score_raw)
                        row['Rerank Score (raw)'] = f"{rerank_raw:.4f}"
                        row['Rerank Score (sigmoid)'] = f"{sigmoid(rerank_raw):.4f}"

                    display_results.append(row)

                df_results = pd.DataFrame(display_results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)

                with st.expander("Detailed Semantic Results"):
                    for i, r in enumerate(results[:5], 1):
                        # support both dict-like and DenseSearchResult objects
                        doc_id = getattr(r, 'doc_id', r.get('DocID') if isinstance(r, dict) else None)
                        rerank_score = getattr(r, 'score', r.get('Score') if isinstance(r, dict) else None)
                        full_text = getattr(r, 'full_text', r.get('Full Text') if isinstance(r, dict) else '')

                        # original FAISS cosine score (if available)
                        orig_cosine = None
                        try:
                            orig_cosine = original_scores.get(doc_id) if 'original_scores' in locals() else None
                        except Exception:
                            orig_cosine = None

                        parts = []
                        if doc_id:
                            parts.append(f"{doc_id}")
                        if orig_cosine is not None:
                            parts.append(f"Cosine: {float(orig_cosine):.4f}")
                        if rerank_score is not None and enable_rerank:
                            rerank_raw = float(rerank_score)
                            rerank_sigmoid = sigmoid(rerank_raw)
                            parts.append(
                                f"Rerank: {rerank_raw:.4f} (sigmoid {rerank_sigmoid:.4f})"
                            )

                        st.markdown(f"**{i}. {' - '.join(parts)}**")
                        st.text(full_text)
                        st.divider()

                st.markdown("---")
                st.subheader("Semantic Scores Chart")

                # Chart metric selection (follow PRD-style behavior)
                rerank_available = enable_rerank and getattr(dense_engine, 'reranker', None) is not None
                chart_options = ['Cosine (FAISS)']
                if rerank_available:
                    chart_options += ['Rerank (raw)', 'Rerank (sigmoid)']

                chart_choice = st.selectbox("Chart metric", chart_options, index=0)

                # PRD-style explanatory label when reranker present
                if rerank_available:
                    st.caption(
                        "PRD: Cross-Encoder outputs raw logit scores (higher = more relevant). "
                        "Use 'Rerank (sigmoid)' to view a 0-1 probability-like mapping."
                    )

                chart_data = []
                for r in results[:10]:
                    if isinstance(r, dict):
                        doc_id = r.get('DocID')
                        score_raw = r.get('Score')
                    else:
                        doc_id = getattr(r, 'doc_id', None)
                        score_raw = getattr(r, 'score', None)

                    if doc_id is None:
                        continue

                    if chart_choice == 'Rerank (raw)':
                        if score_raw is None:
                            continue
                        chart_score = float(score_raw)
                    elif chart_choice == 'Rerank (sigmoid)':
                        if score_raw is None:
                            continue
                        chart_score = sigmoid(float(score_raw)) * 100.0
                    else:  # Cosine (FAISS)
                        cosine = original_scores.get(doc_id)
                        if cosine is None:
                            continue
                        chart_score = map_cosine_to_01(float(cosine)) * 100.0

                    chart_data.append({
                        'Document': doc_id,
                        'Score': chart_score,
                    })

                df_chart = pd.DataFrame(chart_data)
                if not df_chart.empty:
                    st.bar_chart(df_chart.set_index('Document')['Score'])
            else:
                st.warning("No semantic matches found for the query")

    elif sub_menu == "Evaluation":
        st.subheader("Evaluasi Komparatif")

        from evaluation import load_qrels, evaluate_system, save_results_csv, get_summary_table

        # Query input: upload CSV (query_id, query_text) or upload qrels CSV
        uploaded = st.file_uploader("Upload queries CSV (query_id,query_text) OR qrels CSV (query_id,doc_id,relevance_score)", type=['csv'])
        use_sample = st.checkbox("Use sample queries", value=False)

        queries = {}
        uploaded_qrels_used = False
        if uploaded is not None:
            try:
                qdf = pd.read_csv(uploaded, dtype=str)
                cols = [c.lower() for c in qdf.columns]
                # Detect qrels upload
                if 'doc_id' in cols and 'relevance_score' in cols and 'query_id' in cols:
                    # Save uploaded qrels to workspace and load
                    qdf.to_csv('qrels.csv', index=False)
                    st.success('Uploaded file detected as qrels and saved to qrels.csv')
                    uploaded_qrels_used = True
                # Detect queries upload
                elif 'query_id' in cols and 'query_text' in cols:
                    for _, row in qdf.iterrows():
                        qid = str(row['query_id']).strip()
                        qtext = str(row['query_text']).strip()
                        if qid and qtext:
                            queries[qid] = qtext
                    if queries:
                        qdf.to_csv('queries.csv', index=False)
                        st.success(f'Loaded {len(queries)} queries from uploaded file and saved as queries.csv')
                    else:
                        st.error('Uploaded queries CSV contains no valid rows')
                else:
                    st.error('Uploaded CSV not recognized. Provide either queries (query_id,query_text) or qrels (query_id,doc_id,relevance_score)')
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")

        # Load workspace query file if present
        if not queries:
            queries = load_queries_file('queries.csv')
            if queries:
                st.success(f'Loaded {len(queries)} queries from queries.csv')

        if use_sample or (not queries and not uploaded_qrels_used):
            sample_queries = [
                "Kebutuhan energi saat ini meningkat pesat",
                "manajemen energi sistem",
                "efisiensi energi",
            ]
            # map to qrels IDs if available, else use 1..n
            for i, q in enumerate(sample_queries, start=1):
                queries[str(i)] = q

        st.markdown(f"Loaded {len(queries)} queries for evaluation")

        # Save uploaded queries file if detected
        if uploaded is not None and not uploaded_qrels_used and queries:
            try:
                qdf = pd.read_csv(uploaded, dtype=str)
                cols = [c.lower() for c in qdf.columns]
                if 'query_id' in cols and 'query_text' in cols:
                    qdf.to_csv('queries.csv', index=False)
                    st.info('Uploaded query file saved as queries.csv in workspace')
            except Exception:
                pass

        # Load qrels (may have been overwritten by uploaded qrels)
        qrels_graded, qrels_binary = load_qrels('qrels.csv')
        if qrels_graded is None:
            st.error("qrels.csv not found in workspace. Place qrels.csv with columns (query_id,doc_id,relevance_score) or upload it above.")
        else:
            qrels_qids = sorted(set(list(qrels_graded.keys()) + list(qrels_binary.keys())))
            if queries:
                original_query_ids = list(queries.keys())
                queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels_qids}
                invalid_qids = [qid for qid in original_query_ids if qid not in qrels_qids]
                if invalid_qids:
                    st.warning(
                        'Ignored the following query IDs because they are not present in qrels.csv: '
                        + ', '.join(invalid_qids)
                    )

            if not queries:
                queries = {qid: f'Query {qid}' for qid in qrels_qids}
                st.info(
                    'No query texts were available for qrels query IDs. Using qrels query IDs as evaluation labels.'
                )

            st.markdown(
                f"Evaluating exactly {len(queries)} ground-truth queries from qrels.csv: {', '.join(sorted(queries.keys()))}"
            )

            top_k_eval = st.slider('Top-K for evaluation', min_value=1, max_value=20, value=5)
            run_eval = st.button('Run Evaluation')
            if run_eval:
                with st.spinner('Running evaluation for three systems...'):
                    all_results = []

                    # 1) TF-IDF baseline
                    tfidf_calc_local = tfidf_calc  # from earlier calculate_tfidf
                    vsm_calc_local = VSMCalculator(tfidf_calc_local, stemmer, load_stopwords())
                    ranked_tf = {}
                    for qid, qtext in queries.items():
                        hits = vsm_calc_local.search(qtext, threshold=0)
                        ranked_tf[qid] = [doc for doc, _ in hits[:top_k_eval]]
                    res_tfidf = evaluate_system('TF-IDF (VSM)', ranked_tf, qrels_graded, qrels_binary, k=top_k_eval)
                    all_results.append(res_tfidf)

                    # 2) SBERT + FAISS (no rerank)
                    ranked_bi = {}
                    for qid, qtext in queries.items():
                        try:
                            faiss_hits = dense_engine.search(qtext, top_k=top_k_eval)
                            ranked_bi[qid] = [r.doc_id for r in faiss_hits[:top_k_eval]]
                        except Exception:
                            ranked_bi[qid] = []
                    res_bi = evaluate_system('SBERT+FAISS', ranked_bi, qrels_graded, qrels_binary, k=top_k_eval)
                    all_results.append(res_bi)

                    # 3) SBERT + FAISS + Cross-Encoder rerank
                    ranked_ce = {}
                    for qid, qtext in queries.items():
                        try:
                            faiss_hits = dense_engine.search(qtext, top_k=max(top_k_eval * 4, 20))
                            if getattr(dense_engine, 'reranker', None) is not None:
                                reranked = dense_engine.rerank(qtext, faiss_hits)
                                ranked_ce[qid] = [r.doc_id for r in reranked[:top_k_eval]]
                            else:
                                ranked_ce[qid] = [r.doc_id for r in faiss_hits[:top_k_eval]]
                        except Exception:
                            ranked_ce[qid] = []
                    res_ce = evaluate_system('SBERT+FAISS+CrossEncoder', ranked_ce, qrels_graded, qrels_binary, k=top_k_eval)
                    all_results.append(res_ce)

                # Show summary
                st.success('Evaluation completed')
                st.dataframe(get_summary_table(all_results))
            
