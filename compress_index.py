def delta_encode(nums):
    """
    对非空的已排序整数列表进行差值编码。返回的列表中，第一个元素是原始的第一个值，后续元素是差值。
    """
    if not nums:
        return []
    out = [nums[0]]
    for i in range(1, len(nums)):
        out.append(nums[i] - nums[i - 1])
    return out


def front_code_terms(term_list, block_size=4):
    """
    Front-code 将 term_list 排序为块。而返回块的列表。每个块是一个字典：
      { 'first': first_term, 'suffixes': [ {'lcp': int, 'suffix': str}, ... ] }
    term_id 是 term_list 中term的索引（从 0 开始）
    """
    blocks = []
    n = len(term_list)
    for i in range(0, n, block_size):
        block_terms = term_list[i:i + block_size]
        first = block_terms[0]
        suffixes = []
        for t in block_terms[1:]:
            #计算第一个和第t个词的最长公共前缀长度
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
