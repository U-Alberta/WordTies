Code and data for paper: [WordTies: Measuring Word Associations in Language Models via Constrained Sampling](https://aclanthology.org/2022.findings-emnlp.440) (Yao et al., Findings 2022)


Please cite as:
```bibtex
@inproceedings{yao-etal-2022-wordties,
    title = "{W}ord{T}ies: Measuring Word Associations in Language Models via Constrained Sampling",
    author = "Yao, Peiran  and
      Renwick, Tobias  and
      Barbosa, Denilson",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.440",
    pages = "5959--5970"
}

```

Pipeline:

1. **main.py** is used for constrained sampling from MLMs
2. **cooccur.py** counts co-occurrences in sampled sentences.
3. **find-assoc.py** computes conditional probabilities and performs association rule mining.
4. **evaluate.py** calculates prec@k
5. **evaluate-breakdown.py** evaluates prec@k for different types of associations.
6. **evaluate-asymmetry.py** evaluates asymmetric associations.
7. **stat-test.py** performs statistical tests.
8. **link-swow-and-kg.py** Find the shortest paths that links cue and reponse in WordNet and ASCENT++.


In addition, *contextual2static/* folder contains implementations of baselines.
