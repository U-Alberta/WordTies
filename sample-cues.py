import random
with open('data/cues.txt') as f:
    cues = list(f)

with open('data/cues-sample-3k.txt', 'w') as f:
    cues = random.sample(cues, 3000)
    f.writelines(cues)