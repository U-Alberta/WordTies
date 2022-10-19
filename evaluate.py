import json

from tqdm import tqdm
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

task = "bert"

def run_evaluation():
    swow = pd.read_csv("data/swow.csv", sep='\t', dtype={'response': str, 'cue': str})

    with open('data/cues-sample-3k.txt') as f:
        cues = [x.strip() for x in f]

    def read_jsonl(filename):
        topk = {}
        with open(filename) as f:
            for line in f:
                doc = json.loads(line)
                cue = doc['cue']
                cue = cue.lower()
                if cue not in topk:
                    topk[cue] = []
                for assoc in doc['assoc']:
                    token = assoc['word']
                    token = token.lower()
                    topk[cue].append((token, assoc['score']))
        return topk

    topk = read_jsonl(f"results/assocs/{task}.jsonl")

    eval_results = []
    all_gold_scores = []
    all_pred_scores = []
    for cue in tqdm(cues):
        if cue not in topk:
            continue
        doc = topk[cue]
        assoc = [x[0] for x in doc]
        ground_truth = swow[swow["cue"] == cue]
        gold_assocs = set(str(x).lower() for x in ground_truth['response'])
        eval_results.append({'cue': cue})
        for k in [5, 10, 20, 30, 40, 50]:
            prec_at_k = len(set(assoc[:k]) & gold_assocs) / k
            eval_results[-1][f"prec_at_{k}"] = prec_at_k
        
        # spearman correlation
        pred_scores = []
        gold_scores = []
        for w in doc:
            try:
                gold = ground_truth[ground_truth["response"] == w[0]].iloc[0]
                gold_scores.append(gold['R123.Strength'])
                pred_scores.append(w[1])
            except IndexError:
                pass
        if len(pred_scores) > 1:
            r = spearmanr(gold_scores, pred_scores)
            eval_results[-1]['spearman'] = r.correlation
            eval_results[-1]['spearman_p'] = r.pvalue
            all_gold_scores.extend(gold_scores)
            all_pred_scores.extend(pred_scores)

    pooled_eval_results = {}
    spearman = spearmanr(all_gold_scores, all_pred_scores)
    pooled_eval_results['spearman'] = spearman.correlation
    pooled_eval_results['spearman_p'] = spearman.pvalue

    print(spearmanr(all_gold_scores, all_pred_scores))
    for k in [5, 10, 20, 30, 40, 50]:
        xs = [x[f"prec_at_{k}"] for x in eval_results]
        print("prec_at_%d: %.3f" % (k, np.mean(xs)))
        pooled_eval_results[f"prec_at_{k}"] = np.mean(xs)
        

    with open(f'results/prec_at_k/{task}.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    with open(f'results/prec_at_k/pooled-{task}.json', 'w') as f:
        json.dump(pooled_eval_results, f, indent=2)


if __name__ == '__main__':
    for t in ("bert",):
        print("running task", t)
        task = t
        run_evaluation()