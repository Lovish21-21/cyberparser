import argparse 
from pp import load_linking_data, build_queries,build_kb
from training import fri_kb_vectorizer, link_queries, evaluate_linking


def main():
    parser = argparse.ArgumentParser(description="AnnoCTR TECHNIQUE entity linker (TF-IDF + cosine similarity)")
    parser.add_argument("--data", required=True, help="Path to linking JSONL (e.g. train_w_con_w_neg.jsonl)")
    parser.add_argument("--top-k", type=int, default=5, help="How many candidates to retrieve per query")
    args = parser.parse_args()
 
    print(f"Loading linking data from {args.data} ...")
    records = load_linking_data(args.data, entity_type_filter="TECHNIQUE")
    print(f"Loaded {len(records)} TECHNIQUE mentions")
 
    query_texts, gold_label_ids, gold_titles = build_queries(records)
    kb_ids, kb_texts, kb_titles, kb_links = build_kb(records)
    print(f"Built KB with {len(kb_ids)} unique ATT&CK techniques")
 
    print("Fitting TF-IDF over KB...")
    vectorizer, kb_matrix = fit_kb_vectorizer(kb_texts)
 
    print("Linking queries...")
    predictions, scores = link_queries(vectorizer, kb_matrix, kb_ids, query_texts, top_k=args.top_k)
 
    evaluate_linking(predictions, gold_label_ids, k_values=(1, 3, 5))

    print("\n=== Sample Predictions ===")
    for i in range(min(3, len(query_texts))):
        pred_id = predictions[i][0]
        pred_title = kb_titles.get(pred_id, "?")
        print(f"\nMention: {records[i]['mention']}")
        print(f"Gold:      {gold_label_ids[i]} ({gold_titles[i]})")
        print(f"Predicted: {pred_id} ({pred_title}) | score={scores[i][0]:.3f}")
 
    return vectorizer, kb_matrix, kb_ids
 
 
if __name__ == "__main__":
    main()
   