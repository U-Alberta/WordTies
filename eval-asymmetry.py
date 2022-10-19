import json
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr, pearsonr


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


def find_asymmetry_in_swow():
    swow = pd.read_json("data/swow-3k-full.jsonl.gz", lines=True, dtype={'response': str, 'cue': str})
    row_index = {}
    for i, (_, row) in enumerate(swow.iterrows()):
        key = row['cue'], row['response']
        row_index[key] = row
    pairs = zip(swow['cue'], swow['response'])
    asym_pairs = []
    asym_pair_set = set()
    for c, r in tqdm(pairs, total=len(swow)):
        if (r, c) not in row_index:
            continue
        if r < c:
            c, r = r, c
        if (c, r) in asym_pair_set:
            continue
        asym_pair_set.add((c, r))
        forward_row = row_index[(c, r)]
        backward_row = row_index[(r, c)]
        forward_score = forward_row['R123.Strength']
        backward_score = backward_row['R123.Strength']
        asym_score = forward_score / backward_score
        if asym_score < 1:
            asym_score = 1./asym_score
        asym_pairs.append((c, r, forward_score, backward_score, asym_score))
    with open("results/asymmetry/swow-asymmetry.jsonl", "w") as f:
        for c, r, fs, bs, asym_score in asym_pairs:
            f.write(json.dumps({"word1": c, "word2": r, "forward_score": fs, "backward_score": bs, "asymmetry_score": asym_score}) + "\n")


def find_asymmetry_in_assocs(task):
    topk = read_jsonl(f"results/assocs/{task}.jsonl")
    index = {}
    for cue in topk:
        for token, score in topk[cue]:
            index[(cue, token)] = score
    asym_pairs = {}
    for cue, token in index:
        if token < cue:
            cue, token = token, cue
        if (cue, token) in asym_pairs:
            continue
        if (cue, token) in index and (token, cue) in index:
            forward_score = index[(cue, token)]
            backward_score = index[(token, cue)]
            asym_score = forward_score / backward_score
            if asym_score < 1:
                asym_score = 1./asym_score
            asym_pairs[(cue, token)] = (forward_score, backward_score, asym_score)
    with open(f"results/asymmetry/{task}-asymmetry.jsonl", "w") as f:
        rows = list(asym_pairs.items())
        rows.sort(key=lambda x: x[1][2], reverse=True)
        for (cue, token), (fs, bs, asym_score) in rows:
            f.write(json.dumps({"word1": cue, "word2": token, "forward_score": fs, "backward_score": bs, "asymmetry_score": asym_score}) + "\n")


def read_asymmetry_file(filename):
    results = {}
    with open(filename) as f:
        for line in f:
            doc = json.loads(line)
            w1 = doc['word1'].lower()
            w2 = doc['word2'].lower()
            results[w1, w2] = (doc['forward_score'], doc['backward_score'], doc['asymmetry_score'])
    return results


def calc_overlap(task):
    ground_truth = read_asymmetry_file("results/asymmetry/swow-asymmetry.jsonl")
    pred = read_asymmetry_file(f"results/asymmetry/{task}-asymmetry.jsonl")
    gt_scores = []
    pred_scores = []
    wrong = 0
    strong = 0
    for w1, w2 in pred:
        if (w1, w2) in ground_truth:
            pred_scores.append(pred[w1, w2][2])
            gt_scores.append(ground_truth[w1, w2][2])
            if pred_scores[-1] >= 1.1 and gt_scores[-1] >= 1.1:
                strong += 1
        elif (w2, w1) in ground_truth:
            wrong += 1
    print(f"{task} Spearman: {spearmanr(gt_scores, pred_scores)}")
    print(f"{task} Pearson: {pearsonr(gt_scores, pred_scores)}")
    print("Support", len(gt_scores))
    print("Wrong", wrong)
    print("Strong", strong)

if __name__ == '__main__':
    for task in ("bert", "roberta", "distilbert"):
        print(task)
        find_asymmetry_in_assocs(task)
        calc_overlap(task)