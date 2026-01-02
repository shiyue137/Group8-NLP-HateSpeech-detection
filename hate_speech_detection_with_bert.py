import os
import json
import numpy as np
import pandas as pd
import time
import warnings
from collections import Counter

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, precision_score, \
    recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 尝试导入可选库
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    print("⚠️ XGBoost 未安装")
    HAS_XGB = False

# ============================================================
# 1. 加载原始数据
# ============================================================
print("\n" + "=" * 70)
print("步骤 1: 加载原始数据")
print("=" * 70)

ORIGINAL_DATA_PATH = r'F:\工作\上大\NLP\test2\initial datasets\labeled_data.csv'
BASE_FEATURES = r'F:\工作\上大\NLP\test2\feature datasets'
DEPENDENCY_JSON_PATH = r'F:\工作\上大\NLP\test2\feature engineering scripts\dependency_dict.json'

try:
    original_data = pd.read_csv(ORIGINAL_DATA_PATH, encoding='utf-8')
except:
    original_data = pd.read_csv(ORIGINAL_DATA_PATH, encoding='ISO-8859-1')

class_labels = original_data[['tweet', 'class']].copy()
class_labels['index'] = range(len(class_labels))
print(f"✅ 加载 {len(class_labels)} 条推文")

# ============================================================
# 2. 生成 TF-IDF 特征
# ============================================================
print("\n" + "=" * 70)
print("步骤 2: 生成 TF-IDF 特征")
print("=" * 70)

vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    stop_words='english'
)

tfidf_sparse_matrix = vectorizer.fit_transform(class_labels['tweet'])
print(f"✅ TF-IDF 矩阵大小: {tfidf_sparse_matrix.shape}")

tfidf_features = pd.DataFrame(
    tfidf_sparse_matrix.toarray(),
    columns=[f'tfidf:{i}' for i in range(tfidf_sparse_matrix.shape[1])]
)
tfidf_features['index'] = range(len(tfidf_features))

os.makedirs("saved_models", exist_ok=True)
joblib.dump(vectorizer, "saved_models/tfidf_vectorizer.pkl")

# ============================================================
# 2.5. 从 JSON 提取依存特征
# ============================================================
print("\n" + "=" * 70)
print("步骤 2.5: 提取依存特征")
print("=" * 70)

HATE_DICT_PATH = r'F:\工作\上大\NLP\test2\dictionaries\hatebase_dict.csv'

try:
    with open(DEPENDENCY_JSON_PATH, 'r', encoding='utf-8') as f:
        dependency_dict = json.load(f)

    hate_words = set()
    try:
        hate_df = pd.read_csv(HATE_DICT_PATH, encoding='ISO-8859-1', header=None)
        hate_words = set(w.strip().lower() for w in hate_df[0] if isinstance(w, str))
    except:
        hate_words = set()

    selected_deps = ['nsubj', 'dobj', 'amod', 'advmod', 'neg', 'cop', 'compound']
    feature_rows = []

    for idx in range(len(class_labels)):
        dep_list = dependency_dict.get(str(idx), [])
        dep_count = Counter()
        dep_binary = set()
        dep_hate = Counter()

        for item in dep_list:
            rel = item.get("rel")
            head = item.get("head", "")
            dep = item.get("dep", "")

            if rel in selected_deps:
                dep_count[rel] += 1
                dep_binary.add(rel)
                if head in hate_words or dep in hate_words:
                    dep_hate[rel] += 1

        row = {"index": idx}
        for rel in selected_deps:
            row[f"dep_count:{rel}"] = dep_count.get(rel, 0)
            row[f"dep_bin:{rel}"] = 1 if rel in dep_binary else 0
            row[f"dep_hate:{rel}"] = dep_hate.get(rel, 0)
        feature_rows.append(row)

    dependency_features = pd.DataFrame(feature_rows)
    print(f"✅ Dependency features: {dependency_features.shape}")
    feature_status_dependency = True
except Exception as e:
    print(f"⚠️ Dependency 加载失败: {e}")
    dependency_features = None
    feature_status_dependency = False

# ============================================================
# 3. 加载其他特征文件
# ============================================================
print("\n" + "=" * 70)
print("步骤 3: 加载其他特征")
print("=" * 70)

feature_files = {}
feature_status = {}

# 3.1 Weighted TF-IDF
try:
    weighted_tfidf = pd.read_csv(os.path.join(BASE_FEATURES, 'tfidf_scores.csv'))
    score_cols = [col for col in weighted_tfidf.columns if 'score' in col.lower()]
    if score_cols:
        if 'ID' in weighted_tfidf.columns:
            weighted_tfidf = weighted_tfidf[['ID'] + score_cols]
            weighted_tfidf.columns = ['index', 'weighted_TFIDF_scores']
        else:
            weighted_tfidf['index'] = range(len(weighted_tfidf))
            weighted_tfidf = weighted_tfidf[['index', score_cols[0]]]
            weighted_tfidf.columns = ['index', 'weighted_TFIDF_scores']
    feature_files['weighted_tfidf'] = weighted_tfidf
    feature_status['weighted_tfidf'] = True
    print(f"✅ Weighted TF-IDF: {weighted_tfidf.shape}")
except Exception as e:
    print(f"⚠️ Weighted TF-IDF 加载失败: {e}")
    feature_status['weighted_tfidf'] = False

# 3.2 Sentiment
try:
    sentiment_scores = pd.read_csv(os.path.join(BASE_FEATURES, 'sentiment_scores.csv'))
    sentiment_cols = ['hate', 'hatenor', 'neg', 'negnor', 'pos', 'posnor']
    available_cols = [col for col in sentiment_cols if col in sentiment_scores.columns]
    if available_cols:
        if 'index' not in sentiment_scores.columns:
            sentiment_scores['index'] = range(len(sentiment_scores))
        sentiment_scores = sentiment_scores[['index'] + available_cols]
        feature_files['sentiment'] = sentiment_scores
        feature_status['sentiment'] = True
        print(f"✅ Sentiment: {sentiment_scores.shape}")
except Exception as e:
    print(f"⚠️ Sentiment 加载失败: {e}")
    feature_status['sentiment'] = False

# 3.3 Dependency
if feature_status_dependency:
    feature_files['dependency'] = dependency_features
    feature_status['dependency'] = True

# 3.4 Char Bigrams（重点加载）
print("\n🔥 加载 Char Bigrams 特征...")
try:
    char_bigrams = pd.read_csv(os.path.join(BASE_FEATURES, 'char_bigram_features.csv'))
    if 'index' not in char_bigrams.columns:
        char_bigrams['index'] = range(len(char_bigrams))
    char_cols = [col for col in char_bigrams.columns if col.startswith('char_bigrams:')]
    if char_cols:
        char_bigrams = char_bigrams[['index'] + char_cols]
        feature_files['char_bigrams'] = char_bigrams
        feature_status['char_bigrams'] = True
        print(f"✅ Char Bigrams: {char_bigrams.shape} ({len(char_cols)} 维)")
except Exception as e:
    print(f"⚠️ Char Bigrams 加载失败: {e}")
    feature_status['char_bigrams'] = False

# 跳过 Word Bigrams
print("⚠️ 跳过 Word Bigrams")
feature_status['word_bigrams'] = False

# 3.6 TF-IDF
feature_files['tfidf'] = tfidf_features
feature_status['tfidf'] = True

# 3.7 BERT
try:
    bert_features = pd.read_csv(os.path.join(BASE_FEATURES, 'bert_features.csv'))
    if 'index' not in bert_features.columns:
        bert_features['index'] = range(len(bert_features))
    bert_cols = [col for col in bert_features.columns if col.startswith('bert:')]
    if bert_cols:
        bert_features = bert_features[['index'] + bert_cols]
        feature_files['bert'] = bert_features
        feature_status['bert'] = True
        print(f"✅ BERT: {bert_features.shape} ({len(bert_cols)} 维)")
except Exception as e:
    print(f"⚠️ BERT 加载失败: {e}")
    feature_status['bert'] = False

# ============================================================
# 4. 合并特征
# ============================================================
print("\n" + "=" * 70)
print("步骤 4: 合并特征")
print("=" * 70)

master = class_labels.copy()
merge_order = ['weighted_tfidf', 'sentiment', 'dependency', 'char_bigrams', 'tfidf', 'bert']

for feature_name in merge_order:
    if feature_status.get(feature_name, False):
        df_to_merge = feature_files[feature_name]
        master = master.merge(df_to_merge, on='index', how='left')
        print(f"  ✅ 合并 {feature_name}: 总列数 {master.shape[1]}")

master.fillna(0, inplace=True)

print(f"\n✅ 最终特征矩阵: {master.shape}")
print(f"   - 样本数: {master.shape[0]}")
print(f"   - 特征数: {master.shape[1] - 3}")

if feature_status.get('char_bigrams', False):
    char_dim = len([col for col in master.columns if col.startswith('char_bigrams:')])
    print(f"   🔥 包含 Char Bigrams 特征: {char_dim} 维")

# ============================================================
# 5. 准备训练数据
# ============================================================
print("\n" + "=" * 70)
print("步骤 5: 准备训练数据")
print("=" * 70)

y = master['class']
X = master.drop(['index', 'tweet', 'class'], axis=1)

print(f"✅ 特征维度: {X.shape}")
print(f"✅ 标签分布:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ 训练集: {X_train.shape}")
print(f"✅ 测试集: {X_test.shape}")

# 基线特征
if feature_status.get('weighted_tfidf', False):
    x_base = X_train[['weighted_TFIDF_scores']]
    x_base_test = X_test[['weighted_TFIDF_scores']]

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✅ 特征已标准化")

# ============================================================
# 6. 训练模型
# ============================================================
print("\n" + "=" * 70)
print("步骤 6: 训练模型")
print("=" * 70)

start_time = time.time()
results = {}

# 6.1 基线模型
if feature_status.get('weighted_tfidf', False):
    print("\n[1/4] 训练基线模型...")
    lr_base = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42, n_jobs=-1)
    lr_base.fit(x_base, y_train)
    base_pred = lr_base.predict(x_base_test)
    results['Baseline'] = {
        'predictions': base_pred,
        'f1': f1_score(y_test, base_pred, average='micro'),
        'accuracy': accuracy_score(y_test, base_pred),
        'precision': precision_score(y_test, base_pred, average='micro'),
        'recall': recall_score(y_test, base_pred, average='micro')
    }
    print(f"✅ 基线模型 F1: {results['Baseline']['f1']:.4f}")

# 6.2 Logistic Regression
print("\n[2/4] 训练 Logistic Regression...")
t1 = time.time()
lr = LogisticRegression(solver='lbfgs', max_iter=300, random_state=42, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
results['Logistic Regression'] = {
    'predictions': lr_pred,
    'f1': f1_score(y_test, lr_pred, average='micro'),
    'accuracy': accuracy_score(y_test, lr_pred),
    'precision': precision_score(y_test, lr_pred, average='micro'),
    'recall': recall_score(y_test, lr_pred, average='micro')
}
print(f"✅ LR F1: {results['Logistic Regression']['f1']:.4f} (用时: {time.time() - t1:.1f}秒)")

# 6.3 Random Forest
print("\n[3/4] 训练 Random Forest...")
t2 = time.time()
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results['Random Forest'] = {
    'predictions': rf_pred,
    'f1': f1_score(y_test, rf_pred, average='micro'),
    'accuracy': accuracy_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred, average='micro'),
    'recall': recall_score(y_test, rf_pred, average='micro')
}
print(f"✅ RF F1: {results['Random Forest']['f1']:.4f} (用时: {time.time() - t2:.1f}秒)")

# 6.4 XGBoost
if HAS_XGB:
    print("\n[4/4] 训练 XGBoost...")
    t3 = time.time()
    xgb = XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results['XGBoost'] = {
        'predictions': xgb_pred,
        'f1': f1_score(y_test, xgb_pred, average='micro'),
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, average='micro'),
        'recall': recall_score(y_test, xgb_pred, average='micro')
    }
    print(f"✅ XGB F1: {results['XGBoost']['f1']:.4f} (用时: {time.time() - t3:.1f}秒)")

print(f"\n⏱️ 总训练时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 7. 集成模型
# ============================================================
print("\n" + "=" * 70)
print("步骤 7: 训练集成模型")
print("=" * 70)

print("\n训练 Voting Ensemble...")

estimators = [('lr', lr), ('rf', rf)]
if HAS_XGB:
    estimators.append(('xgb', xgb))

voting_ensemble = VotingClassifier(
    estimators=estimators,
    voting='soft',
    weights=[2, 1, 1] if HAS_XGB else [2, 1],
    n_jobs=1
)

voting_ensemble.fit(X_train_scaled, y_train)
voting_pred = voting_ensemble.predict(X_test_scaled)
results['Voting Ensemble'] = {
    'predictions': voting_pred,
    'f1': f1_score(y_test, voting_pred, average='micro'),
    'accuracy': accuracy_score(y_test, voting_pred),
    'precision': precision_score(y_test, voting_pred, average='micro'),
    'recall': recall_score(y_test, voting_pred, average='micro')
}

print(f"✅ Voting Ensemble F1: {results['Voting Ensemble']['f1']:.4f}")

# ============================================================
# 8. 生成可视化图表 - ROC曲线和PR曲线
# ============================================================
print("\n" + "=" * 70)
print("步骤 8: 生成ROC曲线和召回率-精确率曲线")
print("=" * 70)

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

os.makedirs("performance_plots", exist_ok=True)

# 将标签二值化（用于多分类ROC）
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3
class_names = ['Hate Speech', 'Offensive', 'Neither']

# 获取最佳模型
best_model = max(results.keys(), key=lambda k: results[k]['f1'])

# 为每个模型获取预测概率
models_proba = {}

# Baseline
if 'Baseline' in results and feature_status.get('weighted_tfidf', False):
    models_proba['Baseline'] = lr_base.predict_proba(x_base_test)

# Logistic Regression
models_proba['Logistic Regression'] = lr.predict_proba(X_test_scaled)

# Random Forest
models_proba['Random Forest'] = rf.predict_proba(X_test)

# XGBoost
if HAS_XGB:
    models_proba['XGBoost'] = xgb.predict_proba(X_test)

# Voting Ensemble
models_proba['Voting Ensemble'] = voting_ensemble.predict_proba(X_test_scaled)

# ============================================================
# 图1: 多分类ROC曲线（One-vs-Rest）
# ============================================================
print("\n绘制多分类ROC曲线...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
model_list = list(models_proba.keys())

for class_idx in range(n_classes):
    ax = axes[class_idx]

    for i, model_name in enumerate(model_list):
        y_score = models_proba[model_name][:, class_idx]
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})')

    # 绘制对角线（随机猜测）
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title(f'ROC Curve - {class_names[class_idx]}', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('performance_plots/1_roc_curves_multiclass.png', dpi=300, bbox_inches='tight')
print("✅ 保存图1: performance_plots/1_roc_curves_multiclass.png")
plt.close()

# ============================================================
# 图2: 宏平均和微平均ROC曲线
# ============================================================
print("\n绘制宏平均和微平均ROC曲线...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, model_name in enumerate(model_list):
    y_score = models_proba[model_name]

    # 计算微平均ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # 计算宏平均ROC
    fpr_macro = dict()
    tpr_macro = dict()
    roc_auc_macro = dict()
    for i in range(n_classes):
        fpr_macro[i], tpr_macro[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc_macro[i] = auc(fpr_macro[i], tpr_macro[i])

    # 计算所有类别的平均
    all_fpr = np.unique(np.concatenate([fpr_macro[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_macro[i], tpr_macro[i])
    mean_tpr /= n_classes
    roc_auc_macro_avg = auc(all_fpr, mean_tpr)

    # 绘制微平均
    axes[0].plot(fpr_micro, tpr_micro, color=colors[idx % len(colors)], lw=2,
                 label=f'{model_name} (AUC = {roc_auc_micro:.3f})')

    # 绘制宏平均
    axes[1].plot(all_fpr, mean_tpr, color=colors[idx % len(colors)], lw=2,
                 label=f'{model_name} (AUC = {roc_auc_macro_avg:.3f})')

# 对角线
for ax in axes:
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

axes[0].set_title('Micro-Average ROC Curve', fontsize=12, fontweight='bold')
axes[1].set_title('Macro-Average ROC Curve', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('performance_plots/2_roc_curves_averaged.png', dpi=300, bbox_inches='tight')
print("✅ 保存图2: performance_plots/2_roc_curves_averaged.png")
plt.close()

# ============================================================
# 图3: Precision-Recall曲线（每个类别）
# ============================================================
print("\n绘制Precision-Recall曲线...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for class_idx in range(n_classes):
    ax = axes[class_idx]

    for i, model_name in enumerate(model_list):
        y_score = models_proba[model_name][:, class_idx]
        precision, recall, _ = precision_recall_curve(y_test_bin[:, class_idx], y_score)
        avg_precision = average_precision_score(y_test_bin[:, class_idx], y_score)

        ax.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f'{model_name} (AP = {avg_precision:.3f})')

    # 基线（类别比例）
    baseline_precision = np.sum(y_test_bin[:, class_idx]) / len(y_test_bin)
    ax.axhline(y=baseline_precision, color='k', linestyle='--', lw=1,
               label=f'Baseline (AP = {baseline_precision:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax.set_title(f'Precision-Recall Curve - {class_names[class_idx]}', fontsize=12, fontweight='bold')
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('performance_plots/3_precision_recall_curves.png', dpi=300, bbox_inches='tight')
print("✅ 保存图3: performance_plots/3_precision_recall_curves.png")
plt.close()

# ============================================================
# 图4: 微平均和宏平均Precision-Recall曲线
# ============================================================
print("\n绘制平均Precision-Recall曲线...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, model_name in enumerate(model_list):
    y_score = models_proba[model_name]

    # 微平均
    precision_micro, recall_micro, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
    avg_precision_micro = average_precision_score(y_test_bin, y_score, average='micro')

    # 宏平均
    avg_precision_macro = average_precision_score(y_test_bin, y_score, average='macro')

    # 计算每个类别的PR曲线用于宏平均
    precision_dict = dict()
    recall_dict = dict()
    for i in range(n_classes):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])

    # 微平均
    axes[0].plot(recall_micro, precision_micro, color=colors[idx % len(colors)], lw=2,
                 label=f'{model_name} (AP = {avg_precision_micro:.3f})')

    # 宏平均（简化版）
    axes[1].plot([0, 1], [avg_precision_macro, avg_precision_macro],
                 color=colors[idx % len(colors)], lw=2,
                 label=f'{model_name} (AP = {avg_precision_macro:.3f})', linestyle='--')

# 基线
for ax in axes:
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

axes[0].set_title('Micro-Average Precision-Recall Curve', fontsize=12, fontweight='bold')
axes[1].set_title('Macro-Average Precision Scores', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('performance_plots/4_precision_recall_averaged.png', dpi=300, bbox_inches='tight')
print("✅ 保存图4: performance_plots/4_precision_recall_averaged.png")
plt.close()

print(f"\n✅ 所有ROC和PR曲线已保存至 performance_plots/ 文件夹")

# ============================================================
# 9. 保存模型
# ============================================================
print("\n" + "=" * 70)
print("步骤 9: 保存模型")
print("=" * 70)

joblib.dump(voting_ensemble, "saved_models/ensemble_with_char_bigrams.pkl")
joblib.dump(scaler, "saved_models/scaler_with_char_bigrams.pkl")
print("✅ 模型已保存")

feature_config = {
    'feature_dim': X.shape[1],
    'used_weighted_tfidf': feature_status.get('weighted_tfidf', False),
    'used_sentiment': feature_status.get('sentiment', False),
    'used_dependency': feature_status.get('dependency', False),
    'used_char_bigrams': feature_status.get('char_bigrams', False),
    'used_word_bigrams': False,
    'used_tfidf': feature_status.get('tfidf', False),
    'used_bert': feature_status.get('bert', False),
    'feature_names': X.columns.tolist(),
    'requires_scaling': True
}
joblib.dump(feature_config, "saved_models/feature_config_with_char_bigrams.pkl")

performance_report = {
    'results': results,
    'best_model': best_model,
    'feature_status': feature_status,
    'total_features': X.shape[1],
    'char_bigrams_enabled': feature_status.get('char_bigrams', False)
}
joblib.dump(performance_report, "saved_models/performance_report_with_char_bigrams.pkl")

# ============================================================
# 10. 最终报告
# ============================================================
print("\n" + "=" * 70)
print("🎉 训练完成！")
print("=" * 70)

print(f"\n📊 性能指标:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  - F1 Score:  {metrics['f1']:.4f}")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")

if 'Baseline' in results:
    baseline_f1 = results['Baseline']['f1']
    best_f1 = results[best_model]['f1']
    improvement = best_f1 - baseline_f1
    print(f"\n📈 最佳模型 ({best_model}):")
    print(f"   相比基线提升: {improvement:.4f} ({improvement / baseline_f1 * 100:.2f}%)")

print(f"\n🔧 特征使用:")
for name, status in feature_status.items():
    print(f"  {'✅' if status else '❌'} {name}")

print(f"\n📈 总特征维度: {X.shape[1]}")
if feature_status.get('char_bigrams', False):
    char_dim = len([col for col in master.columns if col.startswith('char_bigrams:')])
    print(f"   🔥 Char Bigrams 贡献: {char_dim} 维")

print(f"\n⏱️ 总用时: {time.time() - start_time:.1f}秒")
print(f"\n📁 性能图表已保存至: performance_plots/")
print("=" * 70)