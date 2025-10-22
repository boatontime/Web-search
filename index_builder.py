from build_inverted_index import build_inverted_index
from compress_index import compress_inverted_index
import json


def build_and_save_index(folder_path, index_path_raw, index_path, doc_map_path, skip_interval):
    """Build inverted index from folder_path and save compressed index and doc map to files."""
    result = build_inverted_index(folder_path, skip_interval)
    inverted_index = result.get('inverted_index') if isinstance(result, dict) else result
    doc_id_map = result.get('doc_id_map') if isinstance(result, dict) else None

    # write doc_id_map
    try:
        with open(doc_map_path, 'w', encoding='utf-8') as f:
            json.dump(doc_id_map, f, ensure_ascii=False, indent=2)
        print(f"Success to write {doc_map_path}")
    except Exception as e:
        print(f"Failed to write {doc_map_path}: {e}")

    # 未压缩目录
    try:
        with open(index_path_raw, 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f, ensure_ascii=False, indent=2)
        print(f"Success to write {index_path_raw}")
    except Exception as e:
        print(f"Failed to write {index_path_raw}: {e}")

    # 压缩目录
    try:
        compressed = compress_inverted_index(inverted_index)
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(compressed, f, ensure_ascii=False, indent=2)
        print(f"Success to write {index_path}")
    except Exception as e:
        print(f"Failed to write {index_path}: {e}")

    return index_path, doc_map_path
