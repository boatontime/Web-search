import json
from typing import List, Tuple, Dict, Any

def load_doc_id_map(doc_map_path: str) -> Dict[str, str]:
    """
    从单独的 JSON 文件加载文档ID到文件名的映射。
    
    参数：
      doc_map_path: doc_id_map.json 的路径
    
    返回：
      Dict[str, str]：文档ID（字符串）到文件名的映射
    """
    with open(doc_map_path, encoding='utf-8') as f:
        return json.load(f)


def load_lexicon_and_postings(index_path: str):
    """
    从压缩的倒排索引文件加载词典和倒排列表。
    
    加载并解压缩过程：
    1. 从 index_path 加载压缩的 JSON
    2. 解码前缀编码的词典块（lexicon_blocks）得到完整词项列表
    3. 构建词项到ID的映射（用于快速查找）
    4. 返回 postings_map 供按需解码
    
    参数：
      index_path: inverted_index.json 的路径
    
    返回值：
      (term_list, term_to_id, postings_map)
      - term_list: 完整的词项列表
      - term_to_id: 词项到ID的映射（用于O(1)查找）
      - postings_map: 压缩的倒排列表（按需解码）
    """
    with open(index_path, encoding='utf-8') as f:
        data = json.load(f)

    # 1. 提取压缩数据
    lexicon_blocks = data.get('lexicon_blocks', [])
    postings_map = data.get('postings', {})

    # 2. 解码词典块
    term_list = []
    for block in lexicon_blocks:
        first = block['first']
        term_list.append(first)
        for s in block.get('suffixes', []):
            term_list.append(first[:s['lcp']] + s['suffix'])

    # 3. 构建快速查找映射
    term_to_id = {t: i for i, t in enumerate(term_list)}
    
    return term_list, term_to_id, postings_map


def decode_doc_ids(entry):
    """把 postings entry 的 doc_gaps 差分编码还原为绝对 doc_id 列表。"""
    doc_gaps = entry.get('doc_gaps', [])
    ids: List[int] = []
    acc = 0
    for g in doc_gaps:
        acc += g
        ids.append(acc)
    return ids


def postings_for_term(term, term_to_id, postings_map, cache) -> Tuple[List[int], List[int]]:
    """
    按需解码 postings_map 中给定 term 的压缩条目并返回 (doc_ids_list, skips_list)，
    同时把解码结果写入 cache（key 为 str(term_id)）。
    """
    # term_to_id may map term -> numeric id (for compressed index)
    # or term -> term (string) for raw index; accept either.
    tid = term_to_id.get(term)
    if tid is None:
        return [], []
    key = str(tid)
    # cache key uses the same string key
    if key in cache:
        return cache[key]
    entry = postings_map.get(key)
    if not entry:
        # For compressed index, postings_map keys are numeric-string ids.
        # For raw index, postings_map keys are the term itself (we set term_to_id to map term->term),
        # so if not found return empty.
        cache[key] = ([], [])
        return [], []

    # Detect entry format: compressed entries use 'doc_gaps'; raw entries use 'postings' list
    if 'doc_gaps' in entry:
        # compressed format: decode gaps
        ids = decode_doc_ids(entry)
        skips = entry.get('skips', [])
        cache[key] = (ids, skips)
        return ids, skips
    else:
        # raw format: entry expected to be {"postings": [{"doc_id": id, "positions": [...]}, ...], "skips": [...]}
        postings = entry.get('postings', [])
        ids = [p.get('doc_id') for p in postings]
        # ensure sorted ascending
        ids = sorted(ids)
        skips = entry.get('skips', []) if isinstance(entry.get('skips', []), list) else []
        cache[key] = (ids, skips)
        return ids, skips


def load_raw_lexicon_and_postings(index_raw_path: str):
    """
    Detect index format helper: returns 'raw' or 'compressed' by inspecting file content.
    """


def detect_index_format(index_path: str) -> str:
    """
    Inspect the JSON at index_path and return 'compressed' or 'raw'.
    """
    with open(index_path, encoding='utf-8') as f:
        data = json.load(f)
    # compressed format typically contains top-level keys 'lexicon_blocks' and 'postings'
    if isinstance(data, dict) and ('lexicon_blocks' in data or 'postings' in data):
        return 'compressed'
    # raw format is expected to be a dict mapping term -> { 'postings': [...], 'skips': [...] }
    if isinstance(data, dict):
        # check first value for structure
        for v in data.values():
            if isinstance(v, dict) and 'postings' in v:
                return 'raw'
        # default fallback
        return 'raw'

def load_raw_lexicon_and_postings(index_raw_path: str):
    """
    从未压缩的 inverted_index_raw.json 中加载倒排表，返回 (term_list, term_to_id, postings_map).

    该函数用于支持直接从 build_inverted_index 产生的原始索引检索。

    返回值：
      - term_list: 词项列表（按任意顺序）
      - term_to_id: 将 term 映射为 key（这里我们把 key 设为 term 本身，方便在 postings_map 中查找）
      - postings_map: 与文件中相同的映射 term -> {"postings": [...], "skips": [...]}
    """
    with open(index_raw_path, encoding='utf-8') as f:
        data = json.load(f)

    # build term list from keys of the inverted index
    # data is expected to be { term: {"postings": [...], "skips": [...]}, ... }
    postings_map = data
    term_list = list(postings_map.keys())
    # map term to itself so that postings_for_term() can use str(tid) == term to lookup
    term_to_id = {t: t for t in term_list}
    return term_list, term_to_id, postings_map
