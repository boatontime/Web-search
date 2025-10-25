import os
from parse_xml import parse_xml
from tokenize_and_normalize import tokenize_and_normalize

def build_inverted_index(folder_path, skip_interval):
    """
    基于 parse_xml 和 tokenize_and_normalize 构建倒排表。
    
    参数:
        folder_path: XML 文件路径列表
        skip_interval: 跳表指针间隔设置
            * -1: 不生成跳表指针（所有指针为 -1）
            *  0: 使用 int(sqrt(n)) 的常见启发式设置步长
            *  k: 使用固定步长 k 生成跳表指针
        
    返回:
        inverted_index: dict, 词 -> 文档列表（文档用文件名表示）
    """
    # inverted_index倒排表，记录每个词出现的文档及位置
    inverted_index = {}
    doc_id_map = {}
    _file_to_id = {}
    file_sum = 0

    file_names = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.xml')]
    for idx, file_name in enumerate(file_names, start=1):
        _file_to_id[file_name] = idx
        doc_id_map[idx] = file_name
        file_sum = idx

    percent = 0
    for idx, file_name in enumerate(file_names, start=1):
        file_path = os.path.join(folder_path, file_name)
        if idx / file_sum > percent / 100:
            percent = percent + 1
            print(str(percent) + "%")
        try:
            # 1. 解析 XML
            parsed = parse_xml(file_path)

            # 2. 分词和规范化（忽略 type）
            tokens_dict = tokenize_and_normalize(parsed)

            # 3. 更新倒排表（记录每个词在文档中的位置）
            file_id = _file_to_id[file_name]
            # 遍历当前文档所有分词结果
            pos = 0
            for key, tokens in tokens_dict.items():
                if key == 'type' or not isinstance(tokens, list):
                    continue
                for token in tokens:
                    if token not in inverted_index:
                        inverted_index[token] = {}
                    if file_id not in inverted_index[token]:
                        inverted_index[token][file_id] = []
                    inverted_index[token][file_id].append(pos)
                    pos += 1

        except Exception as e:
            print(f"处理文件 {file_name} 出错: {e}")

    # 4. 将集合转换为有序列表（id 列表）并生成跳表指针
    def build_skips(postings, skip_interval: int):
        """为 postings 列表生成跳表指针。

        参数：
            postings: postings 列表
            skip_interval: 跳步间隔策略
                * -1: 不生成跳表指针（所有指针为 -1）
                *  0: 使用 int(sqrt(n)) 的常见启发式设置步长
                *  k: 使用固定步长 k 生成跳表指针
        
        返回：
            list[int]：与 postings 等长的列表，值为目标下标或 -1（表示无跳转）
        """
        n = len(postings)
        if n <= 1 or skip_interval == -1:
            return [-1] * n

        # 确定步长
        if skip_interval == 0:
            import math
            step = int(math.sqrt(n))
        else:
            step = skip_interval

        # 如果步长太小，不生成跳表
        if step <= 1:
            return [-1] * n
            
        # 生成跳表指针
        skips = [-1] * n
        i = 0
        while i + step < n:
            skips[i] = i + step
            i += step
        return skips

    inverted_with_skips = {}
    for k, doc_pos_map in inverted_index.items():
        postings = []
        for doc_id in sorted(doc_pos_map.keys()):
            positions = doc_pos_map[doc_id]
            postings.append({"doc_id": doc_id, "positions": positions})
        skips = build_skips(postings, skip_interval)
        inverted_with_skips[k] = {"postings": postings, "skips": skips}

    # 返回包含跳表指针的倒排表和文档 id 映射
    return {"inverted_index": inverted_with_skips, "doc_id_map": doc_id_map}
