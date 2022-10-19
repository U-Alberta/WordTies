import json
import pandas as pd
import gzip

swow = pd.read_csv("data/swow.csv", sep='\t', dtype={'response': str, 'cue': str})
swow['lower_cue'] = swow['cue'].apply(lambda x: str(x).lower())

def read_jsonl(filename):
    topk = {}
    with gzip.open(filename, "rt") as f:
        for line in f:
            doc = json.loads(line)
            cue = doc['cue']
            cue = cue.lower()
            if cue not in topk:
                topk[cue] = []
            for assoc in doc['assoc']:
                token = assoc['word']
                token = token.lower()
                topk[cue].append((token, assoc['score'], assoc['top50-sig']))
    return topk

bert = read_jsonl(f"results/assocs/bert-tested.jsonl.gz")
roberta = read_jsonl(f"results/assocs/roberta-tested.jsonl.gz")
distilbert = read_jsonl(f"results/assocs/distilbert-tested.jsonl.gz")

def format_single(w, sig):
    if not sig:
        return "\\textit{" + w + "}"
    else:
        return w

while True:
    cue = input("Enter cue: ")
    cue = cue.lower()
    if cue not in bert:
        print("does not exist")
        continue
    swow_rows = swow[swow['lower_cue'] == cue][:10]
    swow_resps = [(x, True) for x in swow_rows['response']]
    bert_resps = [(x[0], x[2]) for x in bert[cue][:12]]
    roberta_resps = [(x[0], x[2]) for x in roberta[cue][:12]]
    distilbert_resps = [(x[0], x[2]) for x in distilbert[cue][:12]]

    print("swow:")
    print(", ".join(format_single(x[0], x[1]) for x in swow_resps))
    print("bert:")
    print(", ".join(format_single(x[0], x[1]) for x in bert_resps))
    print("roberta:")
    print(", ".join(format_single(x[0], x[1]) for x in roberta_resps))
    print("distilbert:")
    print(", ".join(format_single(x[0], x[1]) for x in distilbert_resps))