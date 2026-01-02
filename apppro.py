import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import re
import torch
from transformers import BertTokenizer, BertModel
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


# ===============================
# 1. 加载模型与资源
# ===============================
@st.cache_resource
def load_resources():
    base = "F:/工作/上大/NLP/test2/saved_models"

    # 加载集成 BERT 的模型
    model = joblib.load(os.path.join(base, "ensemble_with_bert.pkl"))
    vectorizer = joblib.load(os.path.join(base, "tfidf_vectorizer.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler_with_bert.pkl"))
    config = joblib.load(os.path.join(base, "feature_config_with_bert.pkl"))

    # 加载 BERT 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)
    bert_model.eval()

    # 加载仇恨词词典
    hate_words_set = set()
    try:
        # 从训练时保存的 sensitive words
        sensitive_words = joblib.load(os.path.join(base, "sensitive_words.pkl"))
        hate_words_set = set(w.lower() for w in sensitive_words)
    except:
        pass

    # 从 hatebase 词典加载
    try:
        hate_dict_path = "F:/工作/上大/NLP/test2/dictionaries/hatebase_dict.csv"
        hate_df = pd.read_csv(hate_dict_path, encoding='ISO-8859-1', header=None)
        for word in hate_df[0]:
            if isinstance(word, str):
                hate_words_set.add(word.strip("', ").lower())
    except Exception as e:
        print(f"⚠️ 无法加载 hatebase 词典: {e}")

    # 添加常见仇恨词作为后备
    common_hate_words = [
        'nigger', 'nigga', 'faggot', 'fag', 'retard', 'cunt',
        'bitch', 'slut', 'whore', 'bastard', 'damn', 'shit',
        'fuck', 'ass', 'piss', 'pussy', 'cock','dick'
    ]
    hate_words_set.update(common_hate_words)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "config": config,
        "bert_tokenizer": bert_tokenizer,
        "bert_model": bert_model,
        "device": device,
        "sensitive_words": hate_words_set,
        "hate_words": hate_words_set,  # 用于屏蔽
        "feature_names": config['feature_names']
    }


resources = load_resources()


# ===============================
# 2. 文本预处理
# ===============================
def preprocess(text):
    """简单的文本预处理"""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    return text


# ===============================
# 2.5. 仇恨词屏蔽功能 (新增)
# ===============================
def mask_hate_words(text, hate_words_set):
    """
    将文本中的仇恨词用 * 屏蔽

    参数:
        text: 原始文本
        hate_words_set: 仇恨词集合

    返回:
        masked_text: 屏蔽后的文本
        found_words: 找到的仇恨词列表
    """
    words = text.split()
    masked_words = []
    found_words = []

    for word in words:
        # 移除标点符号进行匹配
        clean_word = re.sub(r'[^\w\s]', '', word.lower())

        if clean_word in hate_words_set:
            # 保留首尾字符,中间用*替换
            if len(clean_word) <= 2:
                masked = '*' * len(clean_word)
            else:
                masked = clean_word[0] + '*' * (len(clean_word) - 2) + clean_word[-1]

            # 恢复原始的标点符号
            masked_word = word.lower().replace(clean_word, masked)
            masked_words.append(masked_word)
            found_words.append(clean_word)
        else:
            masked_words.append(word)

    return ' '.join(masked_words), found_words


# ===============================
# 3. 提取 BERT 特征
# ===============================
def extract_bert_features(text, r):
    """提取单条文本的 BERT 特征"""
    encoded = r['bert_tokenizer'](
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(r['device'])
    attention_mask = encoded['attention_mask'].to(r['device'])

    with torch.no_grad():
        outputs = r['bert_model'](input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return cls_embedding.flatten()


# ===============================
# 4. 特征提取 (与训练代码完全一致)
# ===============================
def extract_features(text, r):
    """
    提取特征,必须与训练时的特征顺序和数量完全一致
    """
    feature_names = r['feature_names']
    features = {}

    # 1. Weighted TF-IDF Score
    if any('weighted_TFIDF_scores' in name for name in feature_names):
        words = text.lower().split()
        score = sum(1 for w in words if w in r['sensitive_words'])
        features['weighted_TFIDF_scores'] = score

    # 2. Sentiment Features
    sentiment_cols = ['hate', 'hatenor', 'neg', 'negnor', 'pos', 'posnor']
    if any(col in feature_names for col in sentiment_cols):
        words = text.lower().split()

        # 简化的情感特征计算
        hate_count = sum(1 for w in words if w in r['sensitive_words'])
        hate_ratio = hate_count / len(words) if len(words) > 0 else 0

        # 简单的正负面词检测
        positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'stupid', 'dumb']

        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)

        pos_ratio = pos_count / len(words) if len(words) > 0 else 0
        neg_ratio = neg_count / len(words) if len(words) > 0 else 0

        if 'hate' in feature_names:
            features['hate'] = hate_count
        if 'hatenor' in feature_names:
            features['hatenor'] = hate_ratio
        if 'neg' in feature_names:
            features['neg'] = neg_count
        if 'negnor' in feature_names:
            features['negnor'] = neg_ratio
        if 'pos' in feature_names:
            features['pos'] = pos_count
        if 'posnor' in feature_names:
            features['posnor'] = pos_ratio

    # 3. Dependency Features (占位)
    dep_features = [name for name in feature_names if name.startswith('dep_')]
    for dep_name in dep_features:
        features[dep_name] = 0

    # 4. TF-IDF Features
    tfidf_matrix = r['vectorizer'].transform([text]).toarray()[0]
    for i, val in enumerate(tfidf_matrix):
        col_name = f'tfidf:{i}'
        if col_name in feature_names:
            features[col_name] = val

    # 5. BERT Features (核心新增部分)
    if r['config']['used_bert']:
        bert_features = extract_bert_features(text, r)
        for i, val in enumerate(bert_features):
            col_name = f'bert:{i}'
            if col_name in feature_names:
                features[col_name] = val

    # 按照训练时的特征顺序构建特征向量
    feature_vector = []
    for name in feature_names:
        feature_vector.append(features.get(name, 0))

    X = np.array(feature_vector).reshape(1, -1)

    # 标准化
    X_scaled = r['scaler'].transform(X)
    return X_scaled


# ===============================
# 5. Streamlit UI
# ===============================
st.set_page_config(page_title="Hate Speech Detection", layout="centered")
st.title("🛡️ Hate Speech Detection")
st.caption("基于机器学习模型的仇恨言论检测系统")

# 显示模型信息
with st.expander("ℹ️ 模型信息"):
    st.write(f"特征维度: {resources['config']['feature_dim']}")
    st.write(f"使用的特征类型:")
    st.write(f"- Weighted TF-IDF: {resources['config']['used_weighted_tfidf']}")
    st.write(f"- Sentiment: {resources['config']['used_sentiment']}")
    st.write(f"- Dependency: {resources['config']['used_dependency']}")
    st.write(f"- TF-IDF: {resources['config']['used_tfidf']}")
    st.write(f"- BERT: {resources['config']['used_bert']} ")
    st.write(f"- 仇恨词词典大小: {len(resources['hate_words'])} 个词")

text = st.text_area("请输入英文文本", height=150, placeholder="Enter English text here...")

if st.button("🔍 检测"):
    if not text.strip():
        st.warning("请输入文本")
    else:
        with st.spinner("模型推理中..."):
            try:
                X = extract_features(text, resources)

                # 预测
                try:
                    proba = resources["model"].predict_proba(X)[0]
                    pred = np.argmax(proba)
                except AttributeError:
                    pred = resources["model"].predict(X)[0]
                    proba = np.zeros(3)
                    proba[pred] = 1.0

                # 显示结果
                labels_cn = ["仇恨言论", "攻击性语言", "正常言论"]
                labels_en = ["Hate Speech", "Offensive Language", "Neither"]

                st.subheader("📊 分类结果")

                # 根据预测结果显示不同颜色
                if pred == 0:
                    st.error(f"**{labels_cn[pred]} ({labels_en[pred]})**")
                elif pred == 1:
                    st.warning(f"**{labels_cn[pred]} ({labels_en[pred]})**")
                else:
                    st.success(f"**{labels_cn[pred]} ({labels_en[pred]})**")

                st.subheader("📈 置信度")
                for i, p in enumerate(proba):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{labels_cn[i]} ({labels_en[i]})")
                    with col2:
                        st.write(f"{p:.2%}")
                    st.progress(float(p))

                # ===== 新增: 仇恨词屏蔽功能 =====
                if pred in [0, 1]:  # 如果检测到仇恨言论或攻击性语言
                    st.subheader("🚫 屏蔽后的文本")

                    masked_text, found_words = mask_hate_words(text, resources['hate_words'])

                    if found_words:
                        st.info(masked_text)
                        st.caption(f"检测到 {len(found_words)} 个敏感词并已屏蔽")

                        # 可选: 显示被屏蔽的词(用于调试)
                        with st.expander("🔍 查看被屏蔽的词"):
                            st.write(", ".join(set(found_words)))
                    else:
                        st.info("未检测到仇恨词典中的词汇,但模型识别为不当内容")
                        st.caption("可能包含隐晦表达或上下文不当")

            except Exception as e:
                st.error(f"检测失败: {str(e)}")
                st.exception(e)

