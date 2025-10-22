"""
vector_search.py

提供基于余弦相似度的向量检索（使用之前生成的 doc_vectors.json）。

主要函数：
- search(query: str, vectors_path='doc_vectors.json', top_k=10) -> List[Tuple[str, float]]

实现策略：
- 加载 doc_vectors.json（格式见 doc_vector.py 的输出）
- 建立 term -> list[(doc_id, doc_weight)] 的倒排以便快速累加分数
- 对输入查询使用 tokenize_and_normalize({ 'q': query }) 来得到 tokens
- 计算 query 的 tf-idf（使用 corpus 中 df: 通过倒排长度计算），并做 L2 归一化
- 通过遍历查询项的倒排表累加每个文档的分数（query_weight * doc_weight）
- 返回前 top_k 个文档（file 名称与相似度分数）

注意：该实现假定 doc_vectors 中的向量已经是 L2 归一化的（doc_vector.py 的实现已归一化）。
"""

import json
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from tokenize_and_normalize import tokenize_and_normalize


_cache = {
    'vectors': None,       # dict doc_id -> {'file':..., 'vector':{term:weight}}
    'inv': None,           # inverted index term -> list[(doc_id, doc_weight)]
    'N': None
}


def _load_vectors(vectors_path: str = 'doc_vectors.json') -> Dict[str, Dict]:
    if _cache['vectors'] is None:
        with open(vectors_path, encoding='utf-8') as f:
            _cache['vectors'] = json.load(f)
        # build inverted
        inv = defaultdict(list)
        for doc_id, info in _cache['vectors'].items():
            vec = info.get('vector', {})
            for term, w in vec.items():
                inv[term].append((doc_id, w))
        _cache['inv'] = dict(inv)
        _cache['N'] = len(_cache['vectors'])
    return _cache['vectors']


def search(query: str, vectors_path: str = 'doc_vectors.json', top_k: int = 10) -> List[Tuple[str, float]]:
    """
    对 query 执行向量检索，返回前 top_k 个 (filename, score) 结果（score 为余弦相似度，范围 0..1）。
    """
    vectors = _load_vectors(vectors_path)
    inv = _cache['inv']
    N = _cache['N']

    # tokenize query consistent with document processing
    tokenized = tokenize_and_normalize({'q': query})
    tokens = tokenized.get('q', [])
    if not tokens:
        return []

    q_tf = Counter(tokens)

    # compute idf for query terms by using df from inverted index lengths
    idf = {}
    for term in q_tf.keys():
        df = len(inv.get(term, []))
        # same smoothing as doc_vectors: idf = log((N+1)/(df+1)) + 1
        idf[term] = math.log((N + 1) / (df + 1)) + 1.0

    # build query vector (tf * idf)
    q_vec = {term: (tf * idf.get(term, 0.0)) for term, tf in q_tf.items()}
    # normalize query vector L2
    norm = math.sqrt(sum(v * v for v in q_vec.values()))
    if norm > 0:
        for term in list(q_vec.keys()):
            q_vec[term] = q_vec[term] / norm

    # accumulate scores using inverted lists
    scores = defaultdict(float)
    for term, q_w in q_vec.items():
        postings = inv.get(term, [])
        for doc_id, doc_w in postings:
            scores[doc_id] += q_w * doc_w

    # get top_k docs
    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = [(vectors[doc_id]['file'], score) for doc_id, score in ranked]
    return results


if __name__ == '__main__':
    # simple interactive test
    q = input('query> ')
    res = search(q)
    for fn, sc in res:
        print(f"{sc:.4f}\t{fn}")
