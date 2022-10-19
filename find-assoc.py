from nltk.corpus import stopwords
from tqdm import tqdm
import json
from nltk.corpus import wordnet as wn

task = "bert"
en_stopwords = set(stopwords.words('english'))
counts = {}
word_freq = {}

with open(f"data/{task}-cooccur.tsv") as f:
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
        if cue not in counts:
            counts[cue] = {}
        token = token.lower()
        if token not in counts[cue]:
            counts[cue][token] = 0
        counts[cue][token] += count
        word_freq[token] = word_freq.get(token, 0) + count
word_freq_sum = sum(word_freq.values())

top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
with open(f"data/{task}-top_words.txt", "w") as f:
    for word, freq in top_words[:100]:
        f.write(f"{word}\t{freq}\n")
top_words = set(word for word, freq in top_words[:100])

assocs = {}
for cue in counts:
    z = sum(counts[cue].values())
    for k, v in counts[cue].items():
        counts[cue][k] = v / z
    pairs = sorted(counts[cue].items(), key=lambda x: x[1], reverse=True)
    assocs[cue] = []
    for word, cond_prob in pairs:
        if word in en_stopwords or not word.isalpha() or word in top_words:
            continue
        pmi = cond_prob / word_freq[word]
        if abs(pmi-1) < 0.1:
            print("skipped", word)
            continue
        
        assocs[cue].append((word, cond_prob, pmi))
        if len(assocs[cue]) == 50:
            break
    assocs[cue].sort(key=lambda x: x[1], reverse=True)
topk = assocs

with open(f"results/assocs/{task}.jsonl", "w") as f:
    for cue in topk:
        assocs = [{"word": x[0], "score": x[1]} for x in topk[cue]]
        f.write(json.dumps({"cue": cue, "assoc": assocs}) + "\n")
