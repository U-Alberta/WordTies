import pickle
from collections import Counter
import random

import spacy
from transformers import AutoModel
from tokenizers import Tokenizer as AutoTokenizer
from tqdm import tqdm
import torch
from torchtext.datasets import WikiText103

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task = "albert"
model = "albert-base-v2"

spacy_tokenizer = spacy.load("en_core_web_sm").tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained(model)
bert_tokenizer.enable_truncation(510)

bert_model = AutoModel.from_pretrained(model)
bert_model.cuda()
bert_model.eval()


def c2s():
    with open('../data/wikitext103_word_count.pkl', 'rb') as f:
        full_word_count = pickle.load(f)

    pooled_embeddings = {}
    word_count = Counter()
    for split in ('train', 'valid', 'test'):
        dataset = WikiText103(root='/data/peiran/torchtext', split=split)
        for i, line in tqdm(enumerate(dataset)):
            line: str
            line = line.strip()
            if not line or line.startswith('= = '):
                continue
            words = [x.text for x in spacy_tokenizer(line)]
            word_count.update(words)
            encoding = bert_tokenizer.encode(words, is_pretokenized=True)
            with torch.no_grad():
                input_ids = torch.tensor([encoding.ids], device=device)
                outputs = bert_model(input_ids, output_hidden_states=True).hidden_states
                layer = outputs[1] # output from the first layer, (B, L, N)
                for i, word in enumerate(words):
                    if full_word_count[word] > 1000 and random.random() > 1000/full_word_count[word]:
                        continue
                    span = encoding.word_to_tokens(i)
                    if span is None:
                        continue
                    emb = layer[0, span[0]:span[1], :].mean(dim=0)
                    if word not in pooled_embeddings:
                        pooled_embeddings[word] = emb
                        word_count[word] = 1
                    else:
                        pooled_embeddings[word] += emb
                        word_count[word] += 1

    for w in pooled_embeddings:
        pooled_embeddings[w] /= word_count[w]

    with open(f'../data/c2s_{task}_pooled_embeddings.pkl', 'wb') as f:
        pickle.dump(pooled_embeddings, f)


def vocab():
    with open('../data/cues-sample-3k.txt') as f:
        cues = [x.strip() for x in f]

    pooled_embeddings = {}
    word_count = Counter()
    for cue in tqdm(cues):
        words = [x.text for x in spacy_tokenizer(cue)]
        word_count.update(words)
        encoding = bert_tokenizer.encode(words, is_pretokenized=True, add_special_tokens=False)
        with torch.no_grad():
            input_ids = torch.tensor([encoding.ids], device=device)
            embeddings = bert_model.embeddings.word_embeddings(input_ids)[0].mean(axis=0)
            pooled_embeddings[cue] = embeddings
    for w, wid in bert_tokenizer.get_vocab(with_added_tokens=False).items():
        pooled_embeddings[w] = bert_model.embeddings.word_embeddings.weight[wid].clone().detach()

    with open(f'../data/vocab_{task}_pooled_embeddings.pkl', 'wb') as f:
        pickle.dump(pooled_embeddings, f)


if __name__ == '__main__':
    c2s()
    vocab()