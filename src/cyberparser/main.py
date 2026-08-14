import argparse
from preprocessing import load_data,build_features
from train import train_crf, evaluate_crf

def main():
    parser=argparse.ArgumentParser(description="annoCTR BIO NER parser")
    parser.add_argument("--train",required=True,help="path to training json files")
    parser.add_argument("--dev",required=False, help="path to degv/test file")

    args=parser.parse_args()

    print(f"loading training data")
    train_tokens,train_labels, train_ids = load_bio_data(args.train)
    print(f"loaded")

    print(f"training")
    crf=train_crf(X_train,Y_train)

    if args.dev:
        print(f"\nLoading dev data from {args.dev} ...")
        dev_tokens, dev_labels, dev_ids = load_bio_data(args.dev)
        print(f"Loaded {len(dev_tokens)} dev sentences")
 
        X_dev, y_dev = build_features(dev_tokens, dev_labels)
        evaluate_crf(crf, X_dev, y_dev)
    else:
        print("No --dev file given, skipping evaluation.")
 
    return crf

if __name__ == "__main__":
    main()