import os
import json
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

# -----------------------------
# 1. 连接 CoreNLP Server
# -----------------------------
nlp = StanfordCoreNLP(
    "http://localhost",
    port=9023,
    lang="en",
    timeout=30000
)

# -----------------------------
# 2. 读取数据
# -----------------------------
data_path = r'F:\工作\上大\NLP\test2\initial datasets\labeled_data.csv'
if not os.path.isfile(data_path):
    raise FileNotFoundError(data_path)

data = pd.read_csv(data_path, encoding="ISO-8859-1")

if "index" not in data.columns:
    data["index"] = range(len(data))

# -----------------------------
# 3. 依存句法分析（增强版）
# -----------------------------
dependency_dict = {}

for row in tqdm(data.itertuples(index=False), total=len(data)):
    tweet = str(row.tweet).strip()
    idx = str(row.index)

    if not tweet:
        dependency_dict[idx] = []
        continue

    try:
        parse = nlp.dependency_parse(tweet)
        tokens = nlp.word_tokenize(tweet)

        enhanced_triples = []
        for rel, head_i, dep_i in parse:
            head_word = tokens[head_i - 1] if head_i > 0 and head_i <= len(tokens) else "ROOT"
            dep_word = tokens[dep_i - 1] if dep_i > 0 and dep_i <= len(tokens) else ""

            enhanced_triples.append({
                "rel": rel,
                "head": head_word.lower(),
                "dep": dep_word.lower()
            })

        dependency_dict[idx] = enhanced_triples

    except Exception as e:
        dependency_dict[idx] = []
        print(f"⚠️ Dependency failed at index {idx}: {e}")

# -----------------------------
# 4. 保存 JSON
# -----------------------------
output_path = r'F:\工作\上大\NLP\test2\feature engineering scripts\dependency_dict.json'
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dependency_dict, f, ensure_ascii=False, indent=2)

nlp.close()
print(f"✅ Enhanced dependency parsing saved to:\n{output_path}")
