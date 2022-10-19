import json
import pickle

import torch
import numpy as np
#from pymongo import MongoClient
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel

task = "vocab_albert"
model_name = 'albert-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
#db = MongoClient()['exp']

def fallback_embedding(text):
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    outputs = model(input_ids, output_hidden_states=True).hidden_states
    layer = outputs[1]
    return layer[0, 1:-1].mean(dim=0).cuda()

with open('../data/cues-sample-3k.txt') as f:
    cues = [x.strip() for x in f]

def find_assocs():
    with open(f'../data/{task}_pooled_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
        uncased_embeddings = {}
        with torch.no_grad():
            for k, v in embeddings.items():
                if k.lower() not in uncased_embeddings:
                    uncased_embeddings[k.lower()] = []
                uncased_embeddings[k.lower()].append(v)
            for k, v in uncased_embeddings.items():
                uncased_embeddings[k] = torch.stack(v).mean(dim=0)
        del embeddings
        embeddings = uncased_embeddings
    with open(f"../results/assocs/{task}.jsonl", "w") as f:
        with torch.no_grad():
            all_embs = torch.stack(list(embeddings.values()))
            words = list(embeddings.keys())
            for cue in tqdm(cues):
                if cue not in embeddings:
                    if "c2s" in task:
                        emb = fallback_embedding(cue)
                    else:
                        emb = embeddings[cue.lower()]
                else:
                    emb = embeddings[cue]
                assocs = []
                cosine_sim = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), all_embs)
                topk = torch.topk(cosine_sim, k=50, dim=0)
                for idx, score in zip(topk.indices, topk.values):
                    assocs.append({'word': words[idx], 'score': score.item()})
                f.write(json.dumps({'cue': cue, 'assoc': assocs}) + "\n")

def evaluate():
    swow = pd.read_csv("../data/swow.csv", sep='\t', dtype={'response': str, 'cue': str})
    c2s_assocs = pd.read_json(f"../results/assocs/{task}-assocs.jsonl", lines=True)

    eval_results = []
    all_gold_scores = []
    all_pred_scores = []
    for cue in tqdm(cues):
        doc = None
        try:
            doc = c2s_assocs[c2s_assocs.cue == cue].iloc[0]
        except IndexError:
            print("does not exists", cue)
            continue
        assoc = [x['word'] for x in doc['assoc']]
        ground_truth = swow[swow["cue"] == cue]
        gold_assocs = set(str(x).lower() for x in ground_truth['response'])
        eval_results.append({'cue': cue})
        for k in [5, 10, 20, 30, 40, 50]:
            prec_at_k = len(set(assoc[:k]) & gold_assocs) / k
            eval_results[-1][f"prec_at_{k}"] = prec_at_k
        
        # spearman correlation
        pred_scores = []
        gold_scores = []
        for w in doc['assoc']:
            try:
                gold = ground_truth[ground_truth["response"] == w['word']].iloc[0]
                gold_scores.append(gold['R123.Strength'])
                pred_scores.append(w['score'])
            except IndexError:
                pass
        if len(pred_scores) > 1:
            r = spearmanr(gold_scores, pred_scores)
            eval_results[-1]['spearman'] = r.correlation
            eval_results[-1]['spearman_p'] = r.pvalue
            all_gold_scores.extend(gold_scores)
            all_pred_scores.extend(pred_scores)

    print(spearmanr(all_gold_scores, all_pred_scores))
    for k in [5, 10, 20, 30, 40, 50]:
        xs = [x[f"prec_at_{k}"] for x in eval_results]
        print("prec_at_%d: %.3f" % (k, np.mean(xs)))

    with open(f'../data/{task}_prec_at_k.json', 'w') as f:
        json.dump(eval_results, f, indent=2)


if __name__ == '__main__':
    find_assocs()
