from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import binom
import json
import gzip
from nltk.corpus import wordnet as wn

def load_counts(task):
    print("Load cooccur...")
    en_stopwords = set(stopwords.words('english'))
    counts = {}
    word_freq = {}

    with open(f"data/{task}-cooccur.tsv", "rt") as f:
        for line in tqdm(f):
            cue, token, count = line.strip().split("\t")
            if token in en_stopwords or not token.isalpha():
                continue
            if token == cue:
                continue
            if len(wn.synsets(token.lower())) == 0:
                continue
            if token in ('<unk>', '<s>', '</s>', '<pad>', '<mask>') or "advertisement" in token:
                continue
            count = float(count)
            cue = cue.lower()
            count = float(count)
            if cue not in counts:
                counts[cue] = {}
            token = token.lower()
            if token not in counts[cue]:
                counts[cue][token] = 0
            counts[cue][token] += count
            word_freq[token] = word_freq.get(token, 0) + count
    print("Done")
    return counts, word_freq


def reject_single(c1, c2, alpha=0.1):
    n = int(c1 + c2)
    if n < 2:
        return False
    return c1 > binom.ppf(1-alpha, n, 0.5)


def test(counts, cue, resp, topk=50):
    resp_count = counts[cue].get(resp, 0)
    total_rejects = 0
    total_accepts = 0
    for word, count in counts[cue].items():
        if word == resp:
            continue
        if reject_single(resp_count, count):
            total_rejects += 1
        else:
            total_accepts += 1
        if total_accepts >= topk:
            return False, None
    passed = total_rejects > len(counts[cue]) - 1 - topk
    significant_rank = len(counts[cue]) - 1 - total_rejects
    return passed, significant_rank


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

def test_cue(cue):
    results = []
    for i, assoc in enumerate(topk[cue]):
        passed, significant_rank = test(counts, cue, assoc[0])
        results.append((assoc[0], assoc[1], passed, significant_rank is not None and significant_rank <= i+1, significant_rank))
    return cue, results


def stat_test_ratio(task):
    top_k_sig = {}
    top_k_total = {}
    for k in (5, 10, 20, 30, 40, 50):
        top_k_sig[k] = 0
        top_k_sig[f"{k}_50"] = 0
        top_k_total[k] = 0
        top_k_total[f"{k}_50"] = 0
    with gzip.open(f"results/assocs/{task}-tested.jsonl.gz", "rt") as f:
        for line in f:
            doc = json.loads(line)
            for i,ass in enumerate(doc['assoc']):
                for k in (5, 10, 20, 30, 40, 50):
                    if i > k:
                        continue
                    if ass['significant_rank'] is not None and ass['significant_rank'] <= k:
                        top_k_sig[k] += 1
                    if ass['significant_rank'] is not None:
                        top_k_sig[f"{k}_50"] += 1
                    top_k_total[k] += 1
                    top_k_total[f"{k}_50"] += 1
    for k in top_k_sig:
        print(f"{task} {k} {top_k_sig[k]/top_k_total[k]}")



if __name__ == '__main__':
    """
    for task in ('bert', 'roberta', 'distilbert'):
        counts, _ = load_counts(task)
        topk = read_jsonl(f"results/assocs/{task}.jsonl")
        import multiprocessing as mp
        topk_tested = {}
        with mp.Pool(48) as pool:
            for cue, results in tqdm(pool.imap_unordered(test_cue, topk.keys()), total=len(topk)):
                topk_tested[cue] = results
        topk = topk_tested
        with gzip.open(f"results/assocs/{task}-tested.jsonl.gz", "wt") as f:
            for cue in topk:
                assocs = [
                    {'word': assoc[0], 'score': assoc[1], 'top50-sig': assoc[2], 'topi-sig': assoc[3], 'significant_rank': assoc[4]}
                    for assoc in topk[cue]
                ]
                f.write(json.dumps({'cue': cue, 'assoc': assocs}) + "\n")
    """
    
    for t in ('distilbert', 'bert', 'roberta'):
        stat_test_ratio(t)