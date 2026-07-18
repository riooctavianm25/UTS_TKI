"""Fungsi utilitas preprocessing teks bersama untuk pipeline TF-IDF baseline dan SBERT dense retrieval."""

import re


def case_folding(text):
    text = text.lower()
    text = re.sub(r'\(.*?\)', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenisasi(text):
    return [t for t in text.split() if len(t) >= 3]


def hapus_stopword(tokens, stopwords):
    return [t for t in tokens if t not in stopwords]


def stemming(tokens, stemmer, stopwords):
    hasil = []
    for t in tokens:
        stem = stemmer.stem(t)
        if stem not in stopwords and len(stem) >= 3:
            hasil.append(stem)
    return hasil


def preprocess_tfidf_text(raw_text, stemmer, stopwords):
    folded = case_folding(raw_text)
    tokens = tokenisasi(folded)
    no_stop = hapus_stopword(tokens, stopwords)
    stemmed = stemming(no_stop, stemmer, stopwords)
    return {
        'case_folding': folded,
        'tokens': tokens,
        'no_stopwords': no_stop,
        'stemmed': stemmed,
    }


def preprocess_sbert_text(raw_text):
    return case_folding(raw_text)
