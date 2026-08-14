import json


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
 
 
def sent2labels(labels):
    return list(labels)
 
 
def build_features(all_tokens, all_labels):
    """
    Converts raw token/label lists into CRF-ready X (feature dicts) and y (labels).
    """
    X = [sent2features(sent) for sent in all_tokens]
    y = [sent2labels(labels) for labels in all_labels]
    return X, y




def load_data(filepath):
    all_tokens=[]
    all_labels=[]
    all_ids=[]
    with open(filepath,"r",encoding="utf-8") as f:
        for line_num,line in enumerate(f,1):
            line=line.strip()
            if not line:
                continue
            record=json.loads(line)
            tokens=record.get("tokens")
            lables=record.get("labels")

            if tokens is None or labels is None:
                print(f"[WARN] Missing tokens/all_tags at line {line_num}, id={record.get('id')}")
                continue

            if len(tokens) != len(labels):
                print(f"[WARN] Length mismatch at line {line_num}, id={record.get('id')}: "
                      f"{len(tokens)} tokens vs {len(labels)} labels")
                continue

            all_tokens.append(tokens)
            all_labels.append(labels)
            all_ids.append(record.get("id"))

            
    return all_tokens,all_labels

def word2features(sent, i):
   
    word = sent[i]
 
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.isalpha()": word.isalpha(),
        "word.length": len(word),
    }
 
    if i > 0:
        prev_word = sent[i - 1]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isupper()": prev_word.isupper(),
        })
    else:
        features["BOS"] = True  
 
    if i < len(sent) - 1:
        next_word = sent[i + 1]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isupper()": next_word.isupper(),
        })
    else:
        features["EOS"] = True 
 
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
 
 
def sent2labels(labels):
    return list(labels)
 
 
def build_features(all_tokens, all_labels):
    
    X = [sent2features(sent) for sent in all_tokens]
    y = [sent2labels(labels) for labels in all_labels]
    return X, y

      