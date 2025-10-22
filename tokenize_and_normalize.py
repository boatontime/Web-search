import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def tokenize_and_normalize(data_dict):
    """
    对字典中的字符串字段（忽略 'type'）做英文分词和规范化：
    - 分词
    - 去停用词
    - 小写化
    - 词形还原
    非字符串字段或 type 字段保持原样
    
    参数:
        data_dict: {'type': 'Group', 'who': 'Alice Bob', 'description': 'Leading the team'}
    
    返回:
        dict: 相同键，但字符串字段为规范化词列表
    """
    processed_dict = {}
    
    for key, value in data_dict.items():
        # 忽略 type
        if key == 'type' or not isinstance(value, str):
            processed_dict[key] = value
        else:
            # 分词
            tokens = word_tokenize(value)
            # 仅保留字母，去掉停用词
            tokens = [t for t in tokens if t.isalpha() and t.lower() not in stop_words]
            # 小写化 + 词形还原
            tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens]
            processed_dict[key] = tokens
    
    return processed_dict
