import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 配置
# ============================================================
ORIGINAL_DATA_PATH = r'F:\工作\上大\NLP\test2\initial datasets\labeled_data.csv'
OUTPUT_PATH = r'F:\工作\上大\NLP\test2\feature datasets\bert_features.csv'

# 选择 BERT 模型（可以根据需要更换）
MODEL_NAME = 'bert-base-uncased'  # 或 'bert-large-uncased'
MAX_LENGTH = 128  # 推文最大长度
BATCH_SIZE = 16  # 批处理大小（根据显存调整）

# ============================================================
# 2. 检查 GPU 可用性
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ 使用设备: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 3. 加载 BERT 模型和分词器
# ============================================================
print("\n" + "=" * 70)
print("步骤 1: 加载 BERT 模型")
print("=" * 70)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()  # 设置为评估模式

print(f"✅ BERT 模型加载完成: {MODEL_NAME}")

# ============================================================
# 4. 加载数据
# ============================================================
print("\n" + "=" * 70)
print("步骤 2: 加载原始数据")
print("=" * 70)

try:
    data = pd.read_csv(ORIGINAL_DATA_PATH, encoding='utf-8')
except:
    data = pd.read_csv(ORIGINAL_DATA_PATH, encoding='ISO-8859-1')

if 'index' not in data.columns:
    data['index'] = range(len(data))

print(f"✅ 加载 {len(data)} 条推文")


# ============================================================
# 5. 定义 BERT 特征提取函数
# ============================================================
def extract_bert_features(texts, batch_size=BATCH_SIZE):
    """
    批量提取 BERT 特征

    参数:
        texts: 文本列表
        batch_size: 批处理大小

    返回:
        numpy array: shape (n_samples, 768)
    """
    all_embeddings = []

    # 分批处理
    for i in tqdm(range(0, len(texts), batch_size), desc="提取 BERT 特征"):
        batch_texts = texts[i:i + batch_size]

        # 分词和编码
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        # 将数据移到 GPU
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # 提取特征（不计算梯度）
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 使用 [CLS] token 的输出作为句子表示
            # outputs.last_hidden_state: (batch_size, seq_len, hidden_size)
            # 取第一个 token（[CLS]）的表示
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(cls_embeddings)

    # 合并所有批次
    return np.vstack(all_embeddings)


# ============================================================
# 6. 提取 BERT 特征
# ============================================================
print("\n" + "=" * 70)
print("步骤 3: 提取 BERT 特征")
print("=" * 70)

tweets = data['tweet'].fillna('').tolist()
bert_embeddings = extract_bert_features(tweets)

print(f"✅ BERT 特征维度: {bert_embeddings.shape}")
print(f"   样本数: {bert_embeddings.shape[0]}")
print(f"   特征维度: {bert_embeddings.shape[1]}")

# ============================================================
# 7. 创建 DataFrame 并添加前缀
# ============================================================
print("\n" + "=" * 70)
print("步骤 4: 创建特征 DataFrame")
print("=" * 70)

# 创建列名：bert:0, bert:1, ..., bert:767
bert_columns = [f'bert:{i}' for i in range(bert_embeddings.shape[1])]
bert_df = pd.DataFrame(bert_embeddings, columns=bert_columns)

# 添加 index 列用于合并
bert_df['index'] = data['index'].values

print(f"✅ DataFrame 创建完成: {bert_df.shape}")

# ============================================================
# 8. 保存特征文件
# ============================================================
print("\n" + "=" * 70)
print("步骤 5: 保存 BERT 特征")
print("=" * 70)

bert_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ BERT 特征已保存到: {OUTPUT_PATH}")
print(f"✅ 包含的列: index + {len(bert_columns)} 个 BERT 特征")

# ============================================================
# 9. 特征统计信息
# ============================================================
print("\n" + "=" * 70)
print("特征统计信息")
print("=" * 70)

print(f"\n📊 BERT 特征统计:")
print(f"  - 平均值: {bert_embeddings.mean():.4f}")
print(f"  - 标准差: {bert_embeddings.std():.4f}")
print(f"  - 最小值: {bert_embeddings.min():.4f}")
print(f"  - 最大值: {bert_embeddings.max():.4f}")

print("\n" + "=" * 70)
print("🎉 BERT 特征提取完成！")
print("=" * 70)
print("\n下一步:")
print("  1. 运行 hate_speech_detection.py 训练模型")
print("  2. 模型会自动加载 BERT 特征进行训练")