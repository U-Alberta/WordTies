import json

from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import numpy as np
import pandas as pd
import multiprocessing as mp

task = "roberta"

swow = pd.read_json("data/swow-3k-full.jsonl.gz", lines=True, dtype={'response': str, 'cue': str})

def read_tsv(filename):
    topk = {}
    with open(filename) as f:
        for line in f:
            cue, token, _, __ = line.strip().split("\t")
            token = token.lower()
            cue = cue.lower()
            if cue not in topk:
                topk[cue] = []
            topk[cue].append(token)
    return topk

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
                topk[cue].append(token)
    return topk


def breakdown_cue(cue):
    ks = (5, 10, 20, 30, 40, 50)
    path_breakdown = {}
    source_breakdown = {}
    cue = cue.lower()
    if cue not in topk:
        topk[cue] = []
    ground_truth = swow[swow.cue == cue][:50]
    for i, (_, row) in enumerate(ground_truth.iterrows()):
        if row['prop_path'] is None:
            continue
        path = tuple(row['prop_path'])
        source = row['source']
        if path not in path_breakdown:
            path_breakdown[path] = {"source": source}
            for k in ks:
                path_breakdown[path][k] = {'gold': 0, 'pred': 0}
        if source not in source_breakdown:
            source_breakdown[source] = {}
            for k in ks:
                source_breakdown[source][k] = {'gold': 0, 'pred': 0}
        for k in ks:
            if i < k:
                path_breakdown[path][k]['gold'] += 1
                source_breakdown[source][k]['gold'] += 1

                if row['response'].lower() in topk[cue][:k]:
                    path_breakdown[path][k]['pred'] += 1
                    source_breakdown[source][k]['pred'] += 1
    return path_breakdown, source_breakdown

def breakdown():
    with open('data/cues-sample-3k.txt') as f:
        cues = [x.strip() for x in f]

    path_breakdown = {}
    source_breakdown = {}
    
    with mp.Pool(mp.cpu_count()) as pool:
        for results in tqdm(pool.imap_unordered(breakdown_cue, cues), total=len(cues)):
            path_breakdown_local, source_breakdown_local = results
            for path, breakdown in path_breakdown_local.items():
                if path not in path_breakdown:
                    path_breakdown[path] = breakdown
                else:
                    for k, v in breakdown.items():
                        if k == "source":
                            path_breakdown[path][k] = v
                        else:
                            path_breakdown[path][k]['gold'] += v['gold']
                            path_breakdown[path][k]['pred'] += v['pred']
            for source, breakdown in source_breakdown_local.items():
                if source not in source_breakdown:
                    source_breakdown[source] = breakdown
                else:
                    for k, v in breakdown.items():
                        source_breakdown[source][k]['gold'] += v['gold']
                        source_breakdown[source][k]['pred'] += v['pred']

    for path, breakdown in path_breakdown.items():
        for k, v in breakdown.items():
            if k == "source":
                continue
            if v['gold'] == 0:
                breakdown[k]['accuracy'] = None
            else:
                breakdown[k]['accuracy'] = v['pred'] / v['gold']
    path_breakdown_renamed = {}
    for k, v in path_breakdown.items():
        path_breakdown_renamed["---".join(k)] = v

    for source, breakdown in source_breakdown.items():
        for k, v in breakdown.items():
            if v['gold'] == 0:
                breakdown[k]['accuracy'] = None
            else:
                breakdown[k]['accuracy'] = v['pred'] / v['gold']

    with open(f'results/breakdown/{task}_path.json', 'w') as f:
        json.dump(path_breakdown_renamed, f, indent=2)
    with open(f'results/breakdown/{task}_source.json', 'w') as f:
        json.dump(source_breakdown, f, indent=2)


def path_len_vs_accuracy():
    with open(f'results/breakdown/{task}_path.json') as f:
        path_breakdown = json.load(f)
    lengths = []
    accs = []
    for path, breakdown in path_breakdown.items():
        if breakdown["50"]["accuracy"] is None:
            continue
        lengths.append(len(path.split("---")))
        accs.append(breakdown["50"]["accuracy"])
    print(spearmanr(lengths, accs))
    print(pearsonr(lengths, accs))


if __name__ == '__main__':
    for t in ('bert', 'roberta', 'distilbert', 'c2s_distilbert', 'vocab_bert', 'vocab_distilbert', 'vocab_roberta'):
        task = t
        print(t)
        topk = read_jsonl(f"results/assocs/{task}.jsonl")
        breakdown()
        path_len_vs_accuracy()
