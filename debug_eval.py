from evaluation import load_qrels
from app import load_data
from tfidf_vsm import TFIDFCalculator, VSMCalculator

qg, qb = load_qrels('qrels.csv')
print('qrels_graded:', qg)
print('qrels_binary:', qb)

records, sentences, stemmer, stopwords = load_data()
print('\nnum records:', len(records))
print('sample DocIDs:', [r['DocID'] for r in records[:10]])
record_ids = set(r['DocID'] for r in records)

# qrels referenced docs
all_qrels_docs = set()
for docs in (qg or {}).values():
    all_qrels_docs.update(docs.keys())
print('\nqrels doc ids referenced:', all_qrels_docs)
print('docs missing from records:', sorted(list(all_qrels_docs - record_ids)))
print('docs present in records:', sorted(list(all_qrels_docs & record_ids)))

# Build VSM and show top hits for each qid (map qid->sentence by index if possible)
print('\nBuilding VSM index...')
vc = TFIDFCalculator(records)
vc.build_inverted_index()
vc.calculate_idf()
vc.calculate_tf_idf()
vsm = VSMCalculator(vc, stemmer, stopwords)

for qid in (qg or {}).keys():
    try:
        q_index = int(qid) - 1
        qtext = sentences[q_index]
    except Exception:
        qtext = sentences[0] if sentences else ''
    hits = vsm.search(qtext, threshold=0)
    print(f"\nQuery {qid} -> '{qtext[:80]}...'")
    print('VSM top ids:', [doc for doc, _ in hits[:10]])

print('\nDone.')
