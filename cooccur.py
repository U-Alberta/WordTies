from collections import defaultdict as dd
from gzip import GzipFile
import itertools
import json

from transformers import AutoTokenizer
import spacy
from tqdm import tqdm

task = "bert"
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
spacy_tokenizer = spacy.load("en_core_web_sm").tokenizer

def count_line(line):
    row = json.loads(line)
    cue = row['cue']
    local_count = dd(lambda: 0)
    for prob, w_idx in zip(row['top_k_probs'], row['top_k_idx']):
        row['sent'][row['last_unmask_pos']] = bert_tokenizer.convert_ids_to_tokens(w_idx, skip_special_tokens=True)
        sent = bert_tokenizer.convert_tokens_to_string(row['sent'][1:-1])
        sent = sent.replace("<s>", " ").replace("</s>", " ")
        tokens = [x.text for x in spacy_tokenizer(sent)]
        for token in set(tokens):
            local_count[token] += prob
    local_count = dict(local_count)
    return cue, local_count

if __name__ == '__main__':
    import multiprocessing as mp
    counts = {}
    with mp.Pool(mp.cpu_count()-1) as pool, GzipFile(f"data/{task}.jsonl.gz") as f:
        for cue, local_counts in tqdm(pool.imap_unordered(count_line, itertools.islice(f, 1430*3000), chunksize=32), total=1430*3000):
            if cue not in counts:
                counts[cue] = dd(lambda: 0)
            for token, count in local_counts.items():
                counts[cue][token] += count 
    with open(f"data/{task}-cooccur.tsv", "w") as f:
        for cue, cooccur in counts.items():
            for token, count in cooccur.items():
                f.write(f"{cue}\t{token}\t{count:.5f}\n")