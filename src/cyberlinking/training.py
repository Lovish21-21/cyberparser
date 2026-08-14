from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def fit_kb_vectorizer(kb_texts):
    vectorizer=TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),
    )
    kb_matrix=vectorizer.fit_transform(kb_texts)
    return vectorizer, kb_matrix 

def link_queries(vectorizer,kb_matrix,kb_ids,query_texts,top_k=5):
    query_matrix= vectorizer.transform(query_texts)
    sim_matrix= cosine_similarity(query_matrix,kb_matrix)

    predictions=[]
    scores=[]

    for row in sim_matrix:
        top_idx= np.argsort(row)[::-1][:top_k]
        predictions.append([kb_ids[i] for i in top_idx])
        scores.append([float(row[i]) for i in top_idx])

    return predictions, scores

def evaluate_linking(predictions, gold_label_ids, k_values=(1,3,5)):
    n = len(gold_label_ids)
    results = {}

    for k in k_values:
        correct = 0
        for preds, gold in zip(predictions, gold_label_ids):
            if gold in preds[:k]:
                correct += 1
        results[k] = correct / n if n > 0 else 0.0

    print("\n=== Entity Linking Accuracy ===")
    for k, acc in results.items():
        print(f"Top-{k} accuracy: {acc:.4f}")

    return results