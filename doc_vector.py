"""
doc_vector.py

生成基于 tokenize_and_normalize 输出的文档向量（TF-IDF），并保存为 JSON 文件。

主要函数：
- build_doc_vectors(folder_path, output_path='doc_vectors.json')

输出格式（JSON）：
{
  "1": {"file": "Group 1005141.xml", "vector": {"token1": 0.123, "token2": 0.045, ...}},
  "2": {...},
  ...
}

权重计算：
- idf = log((N + 1) / (df + 1)) + 1  # 平滑并保持正值
- tf = raw term count
- tf-idf = tf * idf
- 向量按 L2 范数归一化

该实现对内存友好程度中等：在内存中保留词典和每个文档的计数。对于非常大的语料建议分批或稀疏化存储。
"""

import os
import json
import math
from collections import Counter, defaultdict
from typing import Dict, List

from parse_xml import parse_xml
from tokenize_and_normalize import tokenize_and_normalize


def build_doc_vectors(folder_path: str, output_path: str = 'doc_vectors.json') -> Dict[str, Dict]:
    """
    从给定文件夹中的 XML 文件生成文档向量（TF-IDF），并将结果保存为 JSON。

    参数：
      folder_path: 包含 XML 文件的文件夹（与 build_inverted_index 使用的相同）
      output_path: 输出 JSON 文件路径（默认 doc_vectors.json）

    返回：
      dict: 已保存的文档向量映射（doc_id -> { 'file': filename, 'vector': { term: weight } })
    """
    # 1) 收集文档列表（保持与 build_inverted_index 相同的编号方式）
    file_names = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.xml')]
    N = len(file_names)

    # mapping doc_id (int starting at 1) -> filename
    doc_id_map: Dict[int, str] = {i: fn for i, fn in enumerate(file_names, start=1)}

    # 2) 为每个文档构建 token 列表并统计 tf；同时统计 df
    docs_terms: Dict[int, List[str]] = {}
    term_df: Dict[str, int] = defaultdict(int)

    print(f"Processing {N} documents to build term statistics...")
    for doc_id, file_name in doc_id_map.items():
        path = os.path.join(folder_path, file_name)
        try:
            parsed = parse_xml(path)
            tokenized = tokenize_and_normalize(parsed)
        except Exception as e:
            print(f"Failed to parse/tokenize {file_name}: {e}")
            tokenized = {}

        # combine tokens from all string fields except 'type'
        tokens: List[str] = []
        for k, v in tokenized.items():
            if k == 'type' or not isinstance(v, list):
                continue
            tokens.extend(v)

        docs_terms[doc_id] = tokens

        # update document frequency: count unique terms per doc
        unique_terms = set(tokens)
        for t in unique_terms:
            term_df[t] += 1

    # 3) compute idf for each term (smoothed)
    idf: Dict[str, float] = {}
    for t, df in term_df.items():
        idf[t] = math.log((N + 1) / (df + 1)) + 1.0

    # 4) compute tf-idf vectors and normalize (L2)
    doc_vectors: Dict[str, Dict] = {}
    for doc_id, tokens in docs_terms.items():
        tf_counts = Counter(tokens)
        vec: Dict[str, float] = {}
        for term, tf in tf_counts.items():
            w = tf * idf.get(term, 0.0)
            vec[term] = w
        # L2 normalization
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for term in list(vec.keys()):
                vec[term] = vec[term] / norm
        # store using string doc_id for JSON keys
        doc_vectors[str(doc_id)] = {'file': doc_id_map[doc_id], 'vector': vec}

    # 5) save to JSON (note: large file possible)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_vectors, f, ensure_ascii=False, indent=2)
        print(f"Saved document vectors to {output_path} (docs={len(doc_vectors)}, terms={len(term_df)})")
    except Exception as e:
        print(f"Failed to save document vectors to {output_path}: {e}")

    return doc_vectors


if __name__ == '__main__':
    # simple CLI: build vectors from DATA folder next to this file
    folder = os.path.join(os.path.dirname(__file__), 'test_datas')
    build_doc_vectors(folder)
