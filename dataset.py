import random

import torch
from torch.utils.data import Dataset

class CondGenDataset(Dataset):
    def __init__(self, filename, trim_to_size=None):
        with open(filename) as f:
            self.data = [x.strip() for x in f]
            self.data = [x for x in self.data if ' ' not in x]
        if trim_to_size:
            self.data = self.data[:trim_to_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CondGenDataLoader:
    def __init__(self, dataset, tokenizer, batch_size=16, min_length=5, max_length=16, shuffle=True):
        """
        a batch = [source, partial, unmask_seq]
        """
        self.queue = [] # queue of batches, actually in a stack
        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)
        self.num_batches = (len(indices) + batch_size - 1) // batch_size

        for i in range(self.num_batches):
            words = [dataset[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
            bsize = len(words)
            length = random.randint(min_length, max_length)
            cue_loc = random.randrange(0, length)
            inputs = []
            masks = torch.ones(bsize, length+2, dtype=torch.bool) # fill mask, 0 if token is a subword of the cue. B * L
            for word_idx, word in enumerate(words):
                seq = [tokenizer.mask_token_id] * cue_loc
                tokens = tokenizer(word, add_special_tokens=False)['input_ids']
                seq.extend(tokens)
                seq.extend([tokenizer.mask_token_id] * (length - len(seq)))
                seq = [tokenizer.cls_token_id] + seq[:length] + [tokenizer.sep_token_id]
                masks[word_idx][cue_loc+1:cue_loc+1+len(tokens)] = 0
                inputs.append(seq)
            self.queue.append({
                'cues': words,
                'cue_loc': cue_loc,
                'input_seqs': inputs,
                'length': length,
                'masks': masks,
            }) 
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.queue) == 0:
            raise StopIteration
        return self.queue.pop()
    
    def __len__(self):
        return len(self.queue)
