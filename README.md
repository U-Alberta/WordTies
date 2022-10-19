Pipeline:

1. **main.py** is used for constrained sampling from MLMs
2. **cooccur.py** counts co-occurrences in sampled sentences.
3. **find-assoc.py** computes conditional probabilities and performs association rule mining.
4. **evaluate.py** calculates prec@k
5. **evaluate-breakdown.py** evaluates prec@k for different types of associations.
6. **evaluate-asymmetry.py** evaluates asymmetric associations.
7. **stat-test.py** performs statistical tests.


In addition, *contextual2static/* folder contains implementations of baselines.