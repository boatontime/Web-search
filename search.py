"""
search.py - 布尔查询解析与执行模块

本模块实现了完整的布尔查询功能，包括：
1. 查询解析：将用户输入的查询字符串解析为抽象语法树（AST）
   - 支持的运算符：&&（与）、||（或）、!（非）
   - 支持括号 () 改变优先级
   - 支持双引号短语（例如 "machine learning"）
   - 运算符优先级：! > && > ||

2. 短语处理：
   - 支持双引号包裹的短语查询，例如 "new york"
   - 短语中的词项必须按顺序相邻出现
   - 单个词也视为短语，统一用 ('PHRASE', [terms]) 表示

3. AST（抽象语法树）节点格式：
   - ('AND', left_node, right_node) - 布尔与
   - ('OR', left_node, right_node)  - 布尔或
   - ('NOT', node)                  - 布尔非
   - ('PHRASE', ['term1', 'term2']) - 短语（词项列表）

4. 查询执行：
   - 使用跳表（skip lists）优化 AND 操作
   - 支持短语匹配（检查词项位置）
   - 结果是按文档ID排序的文件名列表

用法示例：
>>> query = 'algorithm && ("machine learning" || !python)'
>>> ast = parse_boolean_query(query)
>>> results = evaluate_ast(ast)
"""

from typing import List, Tuple, Union, Optional, Dict, Any

from index_loader import load_lexicon_and_postings, decode_doc_ids, postings_for_term


AST = Union[Tuple[str, object], List]  # AST 是嵌套的元组


def tokenize_boolean(query: str) -> List[str]:
	"""
	将查询字符串切分为词符（token）列表。
	
	切分规则：
	1. 短语处理：
	   - 双引号内的内容作为一个完整短语
	   - 保留短语中的空格，例如 "new york" -> ["\"new york\""]
	   - 不处理转义，找到下一个双引号就结束
	
	2. 运算符识别：
	   - &&：逻辑与
	   - ||：逻辑或
	   - !：逻辑非（可以紧跟任何词项或括号）
	
	3. 括号处理：
	   - 左括号 ( 和右括号 ) 分别识别
	   - 用于改变运算优先级
	
	4. 单词识别：
	   - 由字母、数字、下划线或短横线组成
	   - 自动转为小写（大小写不敏感）
	   - 例如：Machine-Learning -> ["machine-learning"]
	
	返回值：
	  List[str]：词符列表，每个词符可能是：
	  - 运算符：'&&', '||', '!'
	  - 括号：'(', ')'
	  - 带引号的短语：'"word1 word2"'
	  - 小写化的单词：'word'
	
	例如：
	>>> tokenize_boolean('ML && ("deep learning" || !python)')
	['ml', '&&', '"deep learning"', '||', '!', 'python']
	"""
	tokens: List[str] = []
	i = 0
	n = len(query)
	while i < n:
		c = query[i]
		if c.isspace():
			i += 1
			continue
		# parentheses
		if c == '(' or c == ')':
			tokens.append(c)
			i += 1
			continue
		# operators: && || !
		if query.startswith('&&', i):
			tokens.append('&&'); i += 2; continue
		if query.startswith('||', i):
			tokens.append('||'); i += 2; continue
		if c == '!':
			tokens.append('!'); i += 1; continue

		# quoted phrase
		if c == '"':
			j = i + 1
			phrase_chars: List[str] = []
			while j < n:
				if query[j] == '"':
					break
				# 接受任何字符直到下一个未转义的引号（不实现转义）
				phrase_chars.append(query[j])
				j += 1
			tokens.append('"' + ''.join(phrase_chars) + '"')
			i = j + 1 if j < n and query[j] == '"' else j
			continue

		# word token
		j = i
		while j < n and (query[j].isalnum() or query[j] in ['_', '-']):
			j += 1
		if j == i:
			# 未识别字符：把它当作单字符 token
			tokens.append(query[i])
			i += 1
		else:
			tokens.append(query[i:j].lower())
			i = j

	return tokens


class _TokenizerStream:
	"""
	词符流包装器类，为递归下降解析器提供基本操作。
	
	主要功能：
	1. 包装 token 列表，提供向前看（peek）和消费（consume）操作
	2. 支持期望值检查，用于验证语法正确性
	3. 跟踪当前处理位置，便于错误报告
	
	工作原理：
	- peek() 返回下一个词符但不移动位置
	- consume() 消费并返回一个词符，可选择性地检查是否匹配期望值
	"""
	def __init__(self, tokens: List[str]):
		"""初始化词符流，接收词符列表作为输入"""
		self.tokens = tokens  # 词符列表
		self.pos = 0         # 当前位置指针

	def peek(self) -> Optional[str]:
		"""
		返回下一个词符但不消费它。
		如果已到达末尾则返回 None。
		"""
		if self.pos < len(self.tokens):
			return self.tokens[self.pos]
		return None

	def consume(self, expected: Optional[str] = None) -> str:
		"""
		消费并返回下一个词符。如果提供了 expected 参数，
		会检查下一个词符是否与期望值匹配。
		
		参数：
		  expected: 期望的词符值（可选）
		
		返回：
		  消费的词符
		
		异常：
		  ValueError: 如果到达输入末尾，或词符与期望值不匹配
		"""
		tk = self.peek()
		if tk is None:
			raise ValueError('查询语法错误：意外的输入结束')
		if expected is not None and tk != expected:
			raise ValueError(f'查询语法错误：期望 {expected}，但得到 {tk}')
		self.pos += 1
		return tk


def _make_phrase_node(token: str) -> Tuple[str, List[str]]:
	"""
	将词符转换为短语节点。
	
	工作流程：
	1. 如果是带引号的短语：
	   - 去除首尾引号
	   - 按空格拆分为词项列表
	   例如：'"new york"' -> ('PHRASE', ['new', 'york'])
	
	2. 如果是普通单词：
	   - 包装为单元素的词项列表
	   例如：'python' -> ('PHRASE', ['python'])
	
	3. 空字符串处理：
	   - 返回空词项列表：('PHRASE', [])
	
	参数：
	  token: 输入词符（可能带引号）
	
	返回：
	  ('PHRASE', terms)，其中 terms 是词项列表
	"""
	if token.startswith('"') and token.endswith('"'):
		inner = token[1:-1].strip()
	else:
		inner = token
	if inner == '':
		terms: List[str] = []
	else:
		# 将短语按空格拆成多个词项；保持小写
		terms = [t for t in inner.split() if t]
	return ('PHRASE', terms)


def parse_boolean_query(query: str) -> AST:
	"""
	将布尔查询字符串解析为抽象语法树（AST）。
	
	语法规则：
	1. 表达式优先级（从高到低）：
	   - 括号 (...)
	   - 逻辑非 !
	   - 逻辑与 &&
	   - 逻辑或 ||
	
	2. 基本元素：
	   - 短语：用双引号包裹，如 "machine learning"
	   - 单词：普通词项，如 python
	   - 复合表达式：用括号组合，如 (a && b)
	
	实现方法：
	使用递归下降解析器，包含以下规则：
	- expr    := or_expr
	- or_expr := and_expr ('||' and_expr)*
	- and_expr := not_expr ('&&' not_expr)*
	- not_expr := '!'* primary
	- primary := '(' expr ')' | PHRASE
	
	返回：
	AST 节点，可能是以下形式之一：
	- ('AND', left_ast, right_ast)  // 与操作
	- ('OR',  left_ast, right_ast)  // 或操作
	- ('NOT', ast)                  // 非操作
	- ('PHRASE', ['word1', 'word2']) // 短语或单词
	
	示例：
	>>> parse_boolean_query('python && ("machine learning" || !django)')
	('AND', 
	  ('PHRASE', ['python']),
	  ('OR',
	    ('PHRASE', ['machine', 'learning']),
	    ('NOT', ('PHRASE', ['django']))))
	
	异常：
	- ValueError: 当查询语法无效时（例如括号不匹配、意外的符号等）
	"""
	tokens = tokenize_boolean(query)
	stream = _TokenizerStream(tokens)

	def parse_expr():
		return parse_or()

	def parse_or():
		node = parse_and()
		while stream.peek() == '||':
			stream.consume('||')
			right = parse_and()
			node = ('OR', node, right)
		return node

	def parse_and():
		node = parse_unary()
		while stream.peek() == '&&':
			stream.consume('&&')
			right = parse_unary()
			node = ('AND', node, right)
		return node

	def parse_unary():
		if stream.peek() == '!':
			stream.consume('!')
			operand = parse_unary()
			return ('NOT', operand)
		return parse_primary()

	def parse_primary():
		tk = stream.peek()
		if tk is None:
			raise ValueError('Unexpected end of input in primary')
		if tk == '(':
			stream.consume('(')
			node = parse_expr()
			if stream.peek() != ')':
				raise ValueError('Missing closing parenthesis')
			stream.consume(')')
			return node
		# phrase or word
		token = stream.consume()
		return _make_phrase_node(token)

	ast = parse_expr()
	if stream.peek() is not None:
		raise ValueError(f'Unexpected token after end of expression: {stream.peek()}')
	return ast

# ------------------ 查询评估与文档搜索 ------------------

def intersect_with_skips(A: List[int], A_skips: List[int], B: List[int], B_skips: List[int]) -> List[int]:
	"""
	使用跳表优化的有序列表交集算法。
	
	算法思路：
	1. 使用两个指针 i, j 分别遍历列表 A 和 B
	2. 当遇到不匹配时，尝试使用跳表指针快速前进
	3. 只有在跳跃不会错过潜在匹配时才跳跃
	
	参数：
	- A, B: 两个升序的文档ID列表
	- A_skips, B_skips: 对应的跳表指针列表
	  - 若 skips[i] = -1 表示位置i没有跳转
	  - 若 skips[i] = k 表示可以从位置i跳到位置k
	
	返回：
	- 升序的交集列表
	
	性能特点：
	- 最佳情况：O(n/√n)，当跳表能有效减少比较次数
	- 最差情况：O(n)，当跳表用不上时退化为普通双指针
	"""
	i = 0
	j = 0
	res: List[int] = []
	lenA = len(A)
	lenB = len(B)
	while i < lenA and j < lenB:
		a = A[i]
		b = B[j]
		if a == b:
			res.append(a)
			i += 1
			j += 1
		elif a < b:
			next_i = A_skips[i] if i < len(A_skips) else -1
			if next_i != -1 and A[next_i] <= b:
				# 多次跳转直到不能保证跳过不会超过匹配项
				while next_i != -1 and A[next_i] <= b:
					i = next_i
					next_i = A_skips[i] if i < len(A_skips) else -1
			else:
				i += 1
		else:
			next_j = B_skips[j] if j < len(B_skips) else -1
			if next_j != -1 and B[next_j] <= a:
				while next_j != -1 and B[next_j] <= a:
					j = next_j
					next_j = B_skips[j] if j < len(B_skips) else -1
			else:
				j += 1
	return res


def intersect_sorted_lists(A: List[int], B: List[int]) -> List[int]:
	"""
	计算两个有序列表的交集（不使用跳表）。
	
	采用双指针算法，时间复杂度 O(min(len(A), len(B)))。
	当跳表不可用或列表较短时使用此函数。
	
	参数：
	- A, B: 升序的文档ID列表
	返回：
	- 升序的交集列表
	"""
	i = 0
	j = 0
	res: List[int] = []
	while i < len(A) and j < len(B):
		if A[i] == B[j]:
			res.append(A[i]); i += 1; j += 1
		elif A[i] < B[j]:
			i += 1
		else:
			j += 1
	return res


def union_sorted_lists(A: List[int], B: List[int]) -> List[int]:
	"""
	计算两个有序列表的并集。
	
	算法特点：
	1. 使用归并排序的思路合并两个有序列表
	2. 重复元素只保留一个
	3. 保持结果有序
	
	参数：
	- A, B: 升序的文档ID列表
	返回：
	- 升序的并集列表
	
	时间复杂度：O(len(A) + len(B))
	空间复杂度：O(len(A) + len(B))
	"""
	i = 0; j = 0; res: List[int] = []
	while i < len(A) and j < len(B):
		if A[i] == B[j]:
			res.append(A[i]); i += 1; j += 1
		elif A[i] < B[j]:
			res.append(A[i]); i += 1
		else:
			res.append(B[j]); j += 1
	while i < len(A): res.append(A[i]); i += 1
	while j < len(B): res.append(B[j]); j += 1
	return res


def difference_lists(universe: List[int], A: List[int]) -> List[int]:
	"""
	计算补集：universe - A（所有在 universe 中但不在 A 中的元素）
	
	主要用于实现 NOT 操作：
	- universe 通常是所有文档ID的列表
	- A 是要排除的文档ID列表
	
	算法：
	1. 同时遍历 universe 和 A
	2. 当遇到 universe[i] = A[j] 时跳过（该元素被排除）
	3. 当 universe[i] < A[j] 时保留 universe[i]（该元素不在排除列表中）
	
	参数：
	- universe: 全集（所有文档ID，升序）
	- A: 要排除的集合（升序）
	
	返回：
	- 补集（升序）
	
	时间复杂度：O(len(universe))
	"""
	res: List[int] = []
	i = 0; j = 0
	while i < len(universe) and j < len(A):
		if universe[i] == A[j]:
			i += 1; j += 1
		elif universe[i] < A[j]:
			res.append(universe[i]); i += 1
		else:
			j += 1
	while i < len(universe): res.append(universe[i]); i += 1
	return res


def _decode_postings_with_positions(entry: Dict[str, Any]) -> Tuple[List[Tuple[int, List[int]]], List[int]]:
	"""
	解码一个压缩的倒排索引项，还原文档ID和词项位置信息。
	
	输入的 entry 结构：
	{
		'doc_gaps': [1,2,2,...],     # 文档ID的差分编码
		'pos_gaps': [[1,5], [2,3]],  # 每个文档中词项位置的差分编码
		'skips': [-1,3,-1,...]       # 跳表指针
	}
	
	解码过程：
	1. 调用 decode_doc_ids 还原文档ID列表
	2. 对每个文档，还原其词项位置列表（差分解码）
	3. 保留跳表指针
	
	返回：
	Tuple[List[Tuple[int, List[int]]], List[int]]
	- 第一个元素：[(doc_id, positions), ...] 列表
	- 第二个元素：跳表指针列表
	"""
	# 支持两种格式：:
	# 1) 压缩格式compress_index: contains 'doc_gaps' and 'pos_gaps'
	# 2) 未压缩格式build_inverted_index: contains 'postings' list with dicts { 'doc_id': id, 'positions': [...] }
	if 'postings' in entry:
		postings_list = entry.get('postings', [])
		postings: List[Tuple[int, List[int]]] = []
		for p in postings_list:
			doc_id = p.get('doc_id')
			positions = p.get('positions', [])
			postings.append((doc_id, positions))
		skips = entry.get('skips', [])
		return postings, skips

	# 没有posting键，默认处理: compressed format
	ids = decode_doc_ids(entry)
	pos_gaps_all = entry.get('pos_gaps', [])
	postings: List[Tuple[int, List[int]]] = []
	for idx, doc_id in enumerate(ids):
		gaps = pos_gaps_all[idx] if idx < len(pos_gaps_all) else []
		positions: List[int] = []
		acc = 0
		for g in gaps:
			acc += g
			positions.append(acc)
		postings.append((doc_id, positions))
	skips = entry.get('skips', [])
	return postings, skips


def _postings_list_to_map(postings: List[Tuple[int, List[int]]]) -> Dict[int, List[int]]:
	return {doc: poses for doc, poses in postings}


def evaluate_ast(ast: AST, index_path: str, doc_id_map_path: str, use_raw: bool = False) -> List[str]:
	"""
	对抽象语法树（AST）进行求值，返回匹配的文件名列表。
	
	主要功能：
	1. 加载并解码压缩的倒排索引
	2. 递归计算布尔表达式
	3. 处理短语查询（检查词项位置）
	4. 使用跳表优化 AND 操作
	
	实现细节：
	1. 缓存机制：
	   - postings_cache: 缓存解码后的 (doc_ids, skips)
	   - postings_pos_cache: 缓存带位置信息的完整 postings
	
	2. 短语处理：
	   - 先找到包含所有词项的文档（使用跳表加速）
	   - 然后验证词项是否按顺序相邻出现
	
	3. 布尔操作：
	   - AND：优先使用跳表交集
	   - OR：有序列表归并
	   - NOT：计算补集
	
	参数：
	- ast: 查询的抽象语法树
	- index_path: 压缩索引文件路径
	- doc_id_map_path: 文档ID映射文件路径
	
	返回：
	- 匹配文档的文件名列表（按文档ID升序）
	
	示例：
	>>> ast = ('AND',
	...     ('PHRASE', ['python']),
	...     ('OR',
	...         ('PHRASE', ['machine', 'learning']),
	...         ('NOT', ('PHRASE', ['django']))))
	>>> results = evaluate_ast(ast)
	"""
	from index_loader import load_lexicon_and_postings, load_doc_id_map, load_raw_lexicon_and_postings, detect_index_format
	# initial load according to use_raw flag
	if use_raw:
		term_list, term_to_id, postings_map = load_raw_lexicon_and_postings(index_path)
	else:
		term_list, term_to_id, postings_map = load_lexicon_and_postings(index_path)
	# If the provided index_path doesn't match use_raw assumption, try to auto-detect and reload.
	try:
		fmt = detect_index_format(index_path)
		if (use_raw and fmt != 'raw') or (not use_raw and fmt != 'compressed'):
			if fmt == 'raw':
				term_list, term_to_id, postings_map = load_raw_lexicon_and_postings(index_path)
			else:
				term_list, term_to_id, postings_map = load_lexicon_and_postings(index_path)
	except Exception:
		# detection 已经是最大努力; 不管error继续使用initial load
		pass
	doc_id_map = load_doc_id_map(doc_id_map_path)

	# caches
	postings_cache: Dict[str, Tuple[List[int], List[int]]] = {}
	postings_pos_cache: Dict[str, Tuple[List[Tuple[int, List[int]]], List[int]]] = {}

	universe_list = sorted(int(k) for k in doc_id_map.keys())

	def eval_node(node) -> Tuple[List[int], Optional[List[int]]]:
		"""
		递归求值 AST 节点，返回匹配的文档ID列表和跳表指针（如果有）。
		
		返回值：Tuple[List[int], Optional[List[int]]]
		- 第一个元素：匹配的文档ID列表（升序）
		- 第二个元素：跳表指针列表，如果不可用则为 None
		
		节点类型处理：
		1. PHRASE 节点：
		   - 单词：直接返回其 postings
		   - 短语：检查词项位置是否相邻
		
		2. NOT 节点：
		   - 计算子节点结果
		   - 返回与全集的差集
		
		3. AND 节点：
		   - 递归计算左右子树
		   - 优先使用跳表求交集
		
		4. OR 节点：
		   - 递归计算左右子树
		   - 合并结果（保持有序）
		"""
		if isinstance(node, tuple) and node and node[0] == 'PHRASE':
			terms = node[1]
			if len(terms) == 0:
				return [], None
			# 对单个词, just return term postings (with skips)
			if len(terms) == 1:
				ids, skips = postings_for_term(terms[0], term_to_id, postings_map, postings_cache)
				return ids, skips
			# 多词短语则需要它们的位置
			# decode full postings with positions for each term
			term_postings_maps: List[Dict[int, List[int]]] = []
			term_doc_lists: List[List[int]] = []
			term_skips_list: List[Optional[List[int]]] = []
			for t in terms:
				tid = term_to_id.get(t)
				key = str(tid) if tid is not None else None
				# 尝试用 postings_for_term to get doc ids and skips
				if key is not None:
					ids, skips = postings_for_term(t, term_to_id, postings_map, postings_cache)
				else:
					ids, skips = [], []
				term_doc_lists.append(ids)
				term_skips_list.append(skips if skips else None)
				# decode positions fully (cache by 'pos_'+key)
				pos_cache_key = f'pos_{key}'
				if key in postings_pos_cache:
					postings_with_pos, _ = postings_pos_cache[key]
				else:
					entry = postings_map.get(key) if key is not None else None
					if entry:
						postings_with_pos, _s = _decode_postings_with_positions(entry)
					else:
						postings_with_pos = []
					postings_pos_cache[key] = (postings_with_pos, [])
				term_postings_maps.append(_postings_list_to_map(postings_pos_cache[key][0]))

			# 嵌入 doc id lists (use skip-aware if both have skips)
			# 从第一个list开始
			candidate = term_doc_lists[0]
			for i in range(1, len(term_doc_lists)):
				B = term_doc_lists[i]
				A_skips = term_skips_list[i-1]
				B_skips = term_skips_list[i]
				if A_skips is not None and B_skips is not None:
					candidate = intersect_with_skips(candidate, A_skips, B, B_skips)
				else:
					candidate = intersect_sorted_lists(candidate, B)

			#为每个文档验证位置相邻性
			matched: List[int] = []
			for doc in candidate:
				positions0 = term_postings_maps[0].get(doc, [])
				pos_sets = [set(term_postings_maps[i].get(doc, [])) for i in range(1, len(terms))]
				found = False
				for p in positions0:
					ok = True
					for offset, s in enumerate(pos_sets, start=1):
						if (p + offset) not in s:
							ok = False
							break
					if ok:
						found = True
						break
				if found:
					matched.append(doc)
			return matched, None

		if isinstance(node, tuple) and node and node[0] == 'NOT':
			sub_ids, _ = eval_node(node[1])
			res = difference_lists(universe_list, sub_ids)
			return res, None

		if isinstance(node, tuple) and node and node[0] == 'AND':
			L, Ls = eval_node(node[1])
			R, Rs = eval_node(node[2])
			if Ls is not None and Rs is not None:
				res = intersect_with_skips(L, Ls, R, Rs)
			else:
				res = intersect_sorted_lists(L, R)
			return res, None

		if isinstance(node, tuple) and node and node[0] == 'OR':
			L, _ = eval_node(node[1])
			R, _ = eval_node(node[2])
			res = union_sorted_lists(L, R)
			return res, None

		if isinstance(node, tuple) and len(node) == 2 and node[0] == 'PHRASE':
			return [], None

		raise ValueError('Unknown AST node in evaluator: %r' % (node,))

	ids, _ = eval_node(ast)
	res = [doc_id_map[str(i)] for i in sorted(ids)]
	return res


