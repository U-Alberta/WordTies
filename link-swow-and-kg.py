import pickle
import csv
from functools import wraps, lru_cache
import os
import gzip
import json
import multiprocessing as mp

try:
    import networkx as nx
    import pandas as pd
    from tqdm import tqdm
    import wn
except ImportError:
    print("Please install the dependencies: pip install wn==0.9.3 tqdm==4.65.0 pandas==1.4.2 networkx==2.7.1")
    exit(1)


wn.download("ewn:2020")
ewn = wn.Wordnet()
os.makedirs("data", exist_ok=True)

# global variables for multiprocessing
_wordnet_G = None
_ascent_G = None


def cached(tmp_dir, name=None):
    # cache the result of a function to a pickle file stored in `tmp_dir`
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal name
            if name is None:
                name = func.__name__
            cache_path = os.path.join(tmp_dir, f"{name}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            else:
                result = func(*args, **kwargs)
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
                return result
        return wrapper
    return decorator


@cached("data", "wordnet-ewn2020-bidirection")
def wordnet_to_nx():
    # export wordnet to a networkx directed graph
    print("Exporting wordnet to a networkx directed graph")

    G = nx.DiGraph()
    for synset in ewn.synsets():
        for rel, targets in synset.relations().items():
            for target in targets:
                G.add_edge(synset.id, target.id, object=rel)
                G.add_edge(target.id, synset.id, object="~" + rel)
        for sense in synset.senses():
            for rel, targets in sense.relations().items():
                for target in targets:
                    G.add_edge(synset.id, target.synset().id, object=rel)
                    G.add_edge(target.synset().id, synset.id, object="~" + rel)
    print("Done: exporting wordnet to a networkx directed graph")
    return G


@cached("data", "ascentpp")
def ascent_to_nx():
    print("Exporting Ascent++ to a networkx directed graph")
    if not os.path.exists("data/ascentpp.csv"):
        print("Downloading ascentpp.csv.tar.gz")
        os.system("wget https://www.mpi-inf.mpg.de/fileadmin/inf/d5/research/ascentpp/ascentpp.csv.tar.gz -O data/ascentpp.csv.tar.gz")
        os.system("tar -xvf data/ascentpp.csv.tar.gz -C data")
        print("Done")

    G = nx.DiGraph()
    with open('data/ascentpp.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            G.add_edge(row[2], row[4], object=row[3])
            G.add_edge(row[4], row[2], object="~" + row[3])
    print("Done: exporting Ascent++ to a networkx directed graph")
    return G


@lru_cache(maxsize=None)
def read_swow():
    if not os.path.exists("data/strength.SWOW-EN.R123.csv"):
        print("Please download `SWOW-EN2008 assoc.strengths (R123)[8Mb]` from https://smallworldofwords.org/en/project/research")
        print("then uncompress and put it in `data/strength.SWOW-EN.R123.csv`")
        exit(1)
    swow = pd.read_csv("data/strength.SWOW-EN.R123.csv", sep="\t")
    swow['cue'] = [str(x) for x in swow['cue']]
    swow['response'] = [str(x) for x in swow['response']]
    return swow


def _find_wordnet_path(args):
    s, t, *_ = args
    G = _wordnet_G
    paths = []
    for s_synset in ewn.synsets(s):
        for t_synset in ewn.synsets(t):
            try:
                path = nx.shortest_path(G, s_synset.id, t_synset.id)
                paths.append(path)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                pass
    if not paths:
        return None
    path = min(paths, key=len)
    prop_path = []
    for i in range(len(path) - 1):
        prop_path.append(G[path[i]][path[i + 1]]['object'])
    return json.dumps({
        'cue': s,
        'response': t,
        'path': path,
        'prop_path': prop_path
    })


def find_wordnet_path():
    print("Finding wordnet paths")
    swow = read_swow()
    global _wordnet_G
    G = wordnet_to_nx()
    _wordnet_G = G

    with gzip.open("data/swow-wordnet.jsonl.gz", "wt") as f, mp.Pool() as pool:
        for line in tqdm(pool.imap_unordered(_find_wordnet_path, swow.values), total=len(swow)):
            if line is not None:
                f.write(line)
                f.write("\n")


def _find_ascent_path_single(args):
    s, t, *_ = args
    G = _ascent_G
    try:
        path = nx.shortest_path(G, s, t)
        prop_path = []
        for i in range(len(path) - 1):
            prop_path.append(G[path[i]][path[i + 1]]['object'])
        return json.dumps({
                'cue': s,
                'response': t,
                'path': path,
                'prop_path': prop_path
            })
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return None


def find_ascent_path():
    print("Finding ascent paths")
    swow = read_swow()
    global _ascent_G
    G = ascent_to_nx()
    _ascent_G = G
    with gzip.open("data/swow-ascent.jsonl.gz", "wt") as f, mp.Pool() as pool:
        for line in tqdm(pool.imap_unordered(_find_ascent_path_single, swow.values), total=len(swow)):
            if line is not None:
                f.write(line)
                f.write("\n")


def merge_ascent_wordnet_paths():
    print("Merging ascent and wordnet paths")
    ascent_df = pd.read_json("data/swow-ascent.jsonl.gz", lines=True)
    wordnet_df = pd.read_json("data/swow-wordnet.jsonl.gz", lines=True)
    wordnet_df.set_index(keys=["cue", "response"], inplace=True)
    ascent_df.set_index(keys=["cue", "response"], inplace=True)
    swow = read_swow()
    merged = []
    for row in tqdm(swow.values, total=len(swow)):
        cue, response = row[:2]
        p1 = None
        p2 = None
        try:
            p1 = wordnet_df.loc[cue, response]["prop_path"]
        except KeyError:
            pass
        try:
            p2 = ascent_df.loc[cue, response]["prop_path"]
        except KeyError:
            pass
        if p1 is None and p2 is None:
            continue
        if p1 is not None and p2 is not None:
            merged.append({"cue": cue, "response": response, "prop_path": min(p1, p2, key=len)})
        elif p1 is not None:
            merged.append({"cue": cue, "response": response, "prop_path": p1})
        elif p2 is not None:
            merged.append({"cue": cue, "response": response, "prop_path": p2})
    merged_df = pd.DataFrame(merged)
    merged_df.to_json("data/swow-full-merged.jsonl.gz", orient="records", lines=True, compression="gzip")


if __name__ == '__main__':
    find_wordnet_path()
    find_ascent_path()
    merge_ascent_wordnet_paths()