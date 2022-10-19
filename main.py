import random
import json
import gzip
import argparse
from contextlib import contextmanager

import torch
from torch.distributions import Categorical
from transformers import AutoModelForMaskedLM, AutoTokenizer, BartForConditionalGeneration, BartTokenizerFast
from tqdm import tqdm

from dataset import CondGenDataset, CondGenDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="generated-v3")
parser.add_argument("--cont", type=bool, default=False)
parser.add_argument('-c', "--compress", default=False, action="store_true")
parser.add_argument("-d", "--decode", default=False, action="store_true")
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--store_top_k', type=int, default=20)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument('--sentence_per_cue', type=int, default=10000)
parser.add_argument('--num_cues', type=int)
parser.add_argument('--mlm_model', type=str, default='bert-base-uncased')
cmd_args = parser.parse_args()

@contextmanager
def open_output_file(*args, **kwargs):
    if cmd_args.compress:
        f = gzip.open(args[0]+".gz", *args[1:], **kwargs)
    else:
        f = open(*args, **kwargs)
    try:
        yield f
    finally:
        f.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)

def sample(p):
    return (p.cumsum(-1) >= torch.rand(p.shape[:-1], device=device).unsqueeze(-1)).byte().argmax(-1)

mlm_model_name = cmd_args.mlm_model
if "bart" in mlm_model_name:
    model = BartForConditionalGeneration.from_pretrained(mlm_model_name, forced_bos_token_id=0).to(device)
    tokenizer = BartTokenizerFast.from_pretrained(mlm_model_name)
else:
    model = AutoModelForMaskedLM.from_pretrained(mlm_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(mlm_model_name)

dataset = CondGenDataset("data/cues-sample-3k.txt", trim_to_size=cmd_args.num_cues)

if not cmd_args.cont:
    with open_output_file(f'data/{cmd_args.task}.jsonl', 'w') as f:
        pass

num_steps = cmd_args.num_steps # mcmc steps
num_epochs = cmd_args.sentence_per_cue # sentence per cue
store_top_k = cmd_args.store_top_k
batch_size = cmd_args.batch_size

model.eval()
for ep in tqdm(range(num_epochs)):
    loader = CondGenDataLoader(dataset, tokenizer, batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(loader):
        input_ids = torch.tensor(batch['input_seqs'], device=device)
        masks = batch['masks'].to(device)
        for step in range(num_steps):
            unmask_pos = random.randrange(batch['length'])
            while unmask_pos == batch['cue_loc']:
                unmask_pos = random.randrange(batch['length'])
            unmask_pos += 1 # the first token is [CLS]
            for seq in batch['input_seqs']:
                seq[unmask_pos] = tokenizer.mask_token_id
            with torch.no_grad():
                logits = model(input_ids).logits[:, unmask_pos]
                # replace mask with predicted token
                probs = logits.softmax(dim=-1)
                tokens = sample(probs)
                mask = masks[:, unmask_pos]
                input_ids[:, unmask_pos].masked_scatter_(mask, tokens)
        
        with torch.no_grad():
            top_ks = probs.topk(k=store_top_k, dim=-1)
        
        with open_output_file(f"data/{cmd_args.task}.jsonl", "at") as f:
            for cue, gen, top_k_p, top_k_i in zip(batch['cues'], input_ids, top_ks.values, top_ks.indices):
                sent = tokenizer.convert_ids_to_tokens(gen)
                probs = []
                idxs = []
                for p, i in zip(top_k_p, top_k_i):
                    if p >= 0.001:
                        probs.append(round(p.item(), 3))
                        idxs.append(i.item())
                    else:
                        break
                
                output = {
                    'cue_loc': batch['cue_loc'],
                    'cue': cue,
                    'sent': sent,
                    'last_unmask_pos': unmask_pos,
                    'top_k_probs': probs,
                    'top_k_idx': idxs
                }
                if cmd_args.decode:
                    output['decoded'] = tokenizer.convert_tokens_to_string(sent)
                
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
