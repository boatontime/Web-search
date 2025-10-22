import sys
from typing import Optional

# keep imports lazy to avoid heavy deps during menu navigation

INDEX_RAW = "inverted_index_raw.json"
INDEX_COMPRESSED = "inverted_index.json"
DOC_MAP = "doc_id_map.json"
DATA_FOLDER = "./test_datas"


def prompt_int(prompt: str, default: Optional[int] = None) -> int:
	while True:
		try:
			s = input(f"{prompt} {('[default: ' + str(default) + ']') if default is not None else ''}: ")
			if s.strip() == "" and default is not None:
				return default
			v = int(s)
			return v
		except ValueError:
			print("请输入一个整数。")


def build_index_interactive():
	try:
		from index_builder import build_and_save_index
	except Exception as e:
		print("无法导入 index_builder：", e)
		return

	print("构建倒排表。请选择跳表指针策略：")
	print("  -1：不生成跳表指针")
	print("   0：使用 int(sqrt(n)) 启发式（默认）")
	print("   k：使用固定步长 k")
	skip = prompt_int("输入 skip_interval", 0)
	print(f"开始构建索引（skip_interval={skip}）……")
	try:
		build_and_save_index(DATA_FOLDER, INDEX_RAW, INDEX_COMPRESSED, DOC_MAP, skip)
		print("构建并保存索引完成。")
	except Exception as e:
		print("构建索引时出错：", e)


def boolean_query_interactive():
	try:
		import search
	except Exception as e:
		print("无法导入 search 模块：", e)
		return

	query = input("请输入布尔查询表达式： ")
	use_raw_input = input("是否使用未压缩索引 (raw)? (y/N): ").strip().lower()
	use_raw = use_raw_input == 'y'

	try:
		ast = search.parse_boolean_query(query)
	except Exception as e:
		print("查询解析失败：", e)
		return

	print("执行查询…… (这可能需要一些时间)")
	try:
		results = search.evaluate_ast(ast, INDEX_RAW if use_raw else INDEX_COMPRESSED, DOC_MAP, use_raw=use_raw)
		if not results:
			print("没有匹配结果。")
		else:
			print(f"找到 {len(results)} 个结果，前 20 个：")
			for fn in results[:20]:
				print(" - ", fn)
	except Exception as e:
		print("查询执行失败：", e)


def build_doc_vectors_interactive():
	try:
		from doc_vector import build_doc_vectors
	except Exception as e:
		print("无法导入 doc_vector：", e)
		return

	print("开始构建文档向量（TF-IDF）并保存为 doc_vectors.json")
	try:
		build_doc_vectors(DATA_FOLDER, 'doc_vectors.json')
		print("文档向量构建完成并保存为 doc_vectors.json")
	except Exception as e:
		print("构建文档向量时出错：", e)


def vector_query_interactive():
	try:
		from vector_search import search as vector_search
	except Exception as e:
		print("无法导入 vector_search：", e)
		return

	q = input("请输入向量查询文本： ")
	try:
		topk = int(input("返回 top-k 结果 (默认 10): ") or 10)
	except ValueError:
		topk = 10

	try:
		results = vector_search(q, 'doc_vectors.json', top_k=topk)
		if not results:
			print("没有匹配结果。")
		else:
			print(f"前 {len(results)} 个匹配（score, filename）：")
			for score, fn in [(sc, f) for f, sc in results]:
				print(f"{score:.4f}\t{fn}")
	except Exception as e:
		print("向量检索失败：", e)


def main_menu():
	while True:
		print("\n=== 倒排索引工具 ===")
		print("1) 构建倒排表（可设置 skip_interval）")
		print("2) 布尔查询（可选择使用未压缩或压缩索引）")
		print("3) 构建文档向量（TF-IDF，保存 doc_vectors.json）")
		print("4) 向量检索（基于 doc_vectors.json 的余弦相似度检索）")
		print("5) 退出")
		choice = input("请选择操作 (1/2/3/4/5): ").strip()
		if choice == '1':
			build_index_interactive()
		elif choice == '2':
			boolean_query_interactive()
		elif choice == '3':
			build_doc_vectors_interactive()
		elif choice == '4':
			vector_query_interactive()
		elif choice == '5':
			print("退出。")
			sys.exit(0)
		else:
			print("无效选择，请重试。")


if __name__ == '__main__':
	main_menu()