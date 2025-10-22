import os
import xml.etree.ElementTree as ET
import re

def parse_xml(file_path):
    """
    按文件名类别解析 XML 文件。
    根节点类型：
        Group*: 根节点 <item>，解析 <who> 和 <description>
        Member*: 根节点 <member>，解析 <self>/<name>
        PastEvent*: 根节点 <item>，解析 <name> 和 <description>
    返回：
        dict 格式
    """
    file_name = os.path.basename(file_path)
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    result = {}

    if file_name.startswith("Group"):
        who = root.findtext("who", default="").strip()
        description = root.findtext("description", default="").strip()
        description = re.sub(r'<[^>]*>', ' ', description)
        description = description.replace('\n', ' ')
        description = re.sub(r'\s+', ' ', description).strip()
        result = {"who": who, "description": description}
        result["type"] = "Group"

    elif file_name.startswith(("Member", "Memeber")):
        name = root.findtext("self/name", default="").strip()
        result = {"name": name}
        result["type"] = "Member"

    elif file_name.startswith("PastEvent"):
        name = root.findtext("name", default="").strip()
        description = root.findtext("description", default="").strip()
        description = re.sub(r'<[^>]*>', ' ', description)
        description = description.replace('\n', ' ')
        description = re.sub(r'\s+', ' ', description).strip()
        result = {"name": name, "description": description}
        result["type"] = "PastEvent"

    else:
        raise ValueError(f"未知文件类型：{file_name}")

    return result
