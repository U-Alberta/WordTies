from torchtext.datasets import WikiText103
from tqdm import tqdm
import requests
import json

def request(method, path, **kwargs):
    url = "https://10.128.0.6:9200" + path
    return requests.request(method, url, auth=('elastic', 'elastic'), verify=False, **kwargs)

def worker(items):
    data = []
    for item in items:
        data.append('{"index": {}}')
        data.append(json.dumps(item))

    r = request("POST", "/wikitext103/_bulk", data="\n".join(data) + "\n", headers={"Content-Type": "application/x-ndjson"})
    assert r.status_code == 200, r.json()

def jobs(split):
    ds = WikiText103(root='/data/torchtext', split=split)
    buf = []
    for i, line in enumerate(ds):
        line: str
        line = line.strip()
        if not line or line.startswith('= = '):
            continue
        buf.append( {
            "split": split,
            "id": i,
            "text": line,
        })
        if len(buf) == 256:
            yield buf
            buf = []
    if buf:
        yield buf


if __name__ == '__main__':
    import multiprocessing as mp
    r = request("DELETE", "/wikitext103")
    with mp.Pool(mp.cpu_count()) as pool:
        for split in ['train', 'valid', 'test']:
            for _ in tqdm(pool.imap(worker, jobs(split))):
                pass
