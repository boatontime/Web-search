def delta_encode(nums):
    """Delta/gap encode a non-empty sorted list of integers.

    Returns list where first element is original first value, subsequent are gaps.
    """
    if not nums:
        return []
    out = [nums[0]]
    for i in range(1, len(nums)):
        out.append(nums[i] - nums[i - 1])
    return out


def front_code_terms(term_list, block_size=4):
    """Front-code sorted term_list into blocks.

    Returns a list of blocks. Each block is a dict:
      { 'first': first_term, 'suffixes': [ {'lcp': int, 'suffix': str}, ... ] }

    term_id is the index of the term in term_list (0-based).
    """
    blocks = []
    n = len(term_list)
    for i in range(0, n, block_size):
        block_terms = term_list[i:i + block_size]
        first = block_terms[0]
        suffixes = []
        for t in block_terms[1:]:
            # compute longest common prefix length between first and t
            lcp = 0
            maxl = min(len(first), len(t))
            while lcp < maxl and first[lcp] == t[lcp]:
                lcp += 1
            suffix = t[lcp:]
            suffixes.append({'lcp': lcp, 'suffix': suffix})
        blocks.append({'first': first, 'suffixes': suffixes})
    return blocks


def compress_inverted_index(inverted_index, block_size=4):
    """压缩倒排索引为紧凑的 JSON 格式。
    
    输入格式 inverted_index:
      term -> { 
        'postings': [ 
          { 'doc_id': int, 'positions': [int,...] }, 
          ... 
        ], 
        'skips': [int,...]  # 跳表指针
      }
    
    压缩策略：
    1. 词典压缩：使用前缀编码将词项分块
    2. 文档ID压缩：差分编码 doc_id 列表
    3. 位置信息压缩：对每个文档的 positions 列表做差分编码
    
    返回字典：
    {
      'lexicon_blocks': [  # 前缀编码的词典块
        {'first': str, 'suffixes': [{'lcp': int, 'suffix': str}, ...]}
      ],
      'postings': {  # 按 term_id 索引的压缩 posting 列表
        tid: {
          'doc_gaps': [...],    # 文档ID的差分编码
          'pos_gaps': [[...]],  # 每个文档的位置差分编码
          'skips': [...]       # 跳表指针（未压缩）
        }
      }
    }
    
    注：doc_id_map（文档ID到文件名的映射）不再包含在压缩索引中，
    而是单独保存在 doc_id_map.json 文件中。
    """
    # 1. 对词项列表排序并前缀编码
    term_list = sorted(inverted_index.keys())
    blocks = front_code_terms(term_list, block_size=block_size)

    # 2. 压缩每个词项的 postings
    postings_map = {}
    for tid, term in enumerate(term_list):
        data = inverted_index[term]
        postings = data.get('postings', [])
        skips = data.get('skips', [])

        # 提取并压缩文档ID
        doc_ids = [p['doc_id'] for p in postings]
        doc_gaps = delta_encode(doc_ids)

        # 提取并压缩每个文档的位置列表
        positions_lists = [p.get('positions', []) for p in postings]
        pos_gaps = [delta_encode(pl) for pl in positions_lists]

        # 存储压缩后的数据
        postings_map[str(tid)] = {
            'doc_gaps': doc_gaps,    # 文档ID差分编码
            'pos_gaps': pos_gaps,    # 位置差分编码
            'skips': skips          # 跳表指针（保持不变）
        }

    return {
        'lexicon_blocks': blocks,    # 前缀编码的词典
        'postings': postings_map     # 压缩的倒排列表
    }
