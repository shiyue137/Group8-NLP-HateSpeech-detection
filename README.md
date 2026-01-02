# 🛡️ Group 8: CivilityAI - 仇恨言论检测系统 (Hate Speech Detection System)

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---
[English](#-group-8-civilityai---hate-speech-detection-system-english) | [简体中文](#-group-8-civilityai---仇恨言论检测系统-简体中文)
---

<a name="-group-8-civilityai---hate-speech-detection-system-english"></a>
## 📖 Project Overview (English)

**CivilityAI** is an advanced NLP-based system designed to detect and mitigate hate speech in online text. Developed by **Group 8** for our NLP course project, this system leverages a hybrid approach combining traditional linguistic features and state-of-the-art **BERT embeddings**.

The system not only classifies text into **Hate Speech**, **Offensive Language**, or **Neutral**, but also provides a real-time visualization of confidence scores and an automatic **hate word masking** feature to foster a safer online environment.

### ✨ Key Features

- **🚀 Hybrid Model Architecture**: 
  - Integrates **BERT Contextual Embeddings** for deep semantic understanding.
  - Utilizes **TF-IDF & Weighted TF-IDF** for keyword importance.
  - Incorporates **Sentiment Analysis** & **Dependency Parsing** features.
  - **Ensemble Voting Classifier** (Logistic Regression + Random Forest + XGBoost) for robust predictions.

- **📊 Comprehensive Analysis**:
  - Classification: `Hate Speech` | `Offensive` | `Neither`.
  - Confidence score visualization for each category.

- **🛡️ Smart Mitigation**:
  - **Auto-Masking**: Automatically detects and censors explicit hate terms.
  - **Sensitive Word Dictionary**: Built-in and extensible dictionary for keyword filtering.

- **🖥️ Interactive UI**:
  - User-friendly web interface built with **Streamlit**.

---

<a name="-group-8-civilityai---仇恨言论检测系统-简体中文"></a>
## 📖 项目简介 (简体中文)

**CivilityAI** 是一个基于 NLP 技术的仇恨言论检测与缓解系统。本项目由 **第八组 (Group 8)** 开发，作为自然语言处理课程作业，采用了结合传统语言学特征与 **BERT** 深度学习特征的混合模型方法。

该系统不仅能够将文本分类为 **仇恨言论**、**攻击性语言** 或 **正常言论**，还提供实时的置信度可视化，并具备自动 **仇恨词屏蔽** 功能，致力于构建更安全的网络环境。

### ✨ 核心功能

- **🚀 混合模型架构**: 
  - 集成 **BERT 上下文嵌入**，深度理解语义。
  - 利用 **TF-IDF & 加权 TF-IDF** 捕捉关键词特征。
  - 融合 **情感分析** 与 **依存句法分析** 特征。
  - **投票集成分类器 (Voting Ensemble)**: 结合逻辑回归、随机森林和 XGBoost，提供稳健的预测能力。

- **📊 全面分析**:
  - 多分类检测: `仇恨言论` | `攻击性语言` | `正常言论`。
  - 为每个类别提供详细的置信度评分可视化。

- **🛡️ 智能防护**:
  - **自动屏蔽**: 自动识别并打码显式仇恨词汇（如 `h*te`）。
  - **敏感词词典**: 内置可扩展的关键词过滤词典。

- **🖥️ 交互式界面**:
  - 基于 **Streamlit** 构建的现代化 Web 界面，操作简便。

---

## 🛠️ Technology Stack / 技术栈

| Component / 组件 | Technology / 技术 | Description / 说明 |
|------------------|-------------------|-------------------|
| **Core / 核心** | Python | Primary programming language / 主要编程语言 |
| **NLP** | Transformers (BERT) | Feature extraction / 特征提取 |
| **ML / 机器学习** | Scikit-learn, XGBoost | Classifiers / 分类器 |
| **Frontend / 前端** | Streamlit | Web Application / Web 应用界面 |
| **Data / 数据** | Pandas, NumPy | Data processing / 数据处理 |
| **Viz / 可视化** | Matplotlib, Seaborn | Performance plots / 性能图表 |

## 🚀 Getting Started / 快速开始

### 1. Installation / 安装

```bash
# Clone the repository / 克隆仓库
git clone https://github.com/shiyue137/Group8-NLP-HateSpeech-detection.git
cd Group8-NLP-HateSpeech-detection

# Install dependencies / 安装依赖
pip install -r requirements.txt
```

### 2. Resources Setup / 资源准备
Ensure model files are in `saved_models/` and dictionaries in `dictionaries/`.
请确保模型文件位于 `saved_models/` 目录，词典文件位于 `dictionaries/` 目录。

### 3. Usage / 使用方法

**Run the Web Application / 启动 Web 应用:**
```bash
streamlit run apppro.py
```

**Retrain the Model / 重新训练模型:**
```bash
python hate_speech_detection_with_bert.py
```

## 👥 Team - Group 8 / 第八组成员

*   **Member 1**
*   **Member 2**
*   **Member 3**
*   **Member 4**

## 🙏 Acknowledgements / 致谢

This project was inspired by and built upon the open-source work of **[tpawelski/hate-speech-detection](https://github.com/tpawelski/hate-speech-detection)**. We have extended the original work by integrating BERT embeddings, developing a web interface, and enhancing the feature engineering process.

Data sources include:
*   **Hatebase.org**: For the initial hate speech dictionary.
*   **Jeffrey Breen's Twitter Sentiment Analysis**: For positive/negative sentiment lexicons.

本项目借鉴并基于开源项目 **[tpawelski/hate-speech-detection](https://github.com/tpawelski/hate-speech-detection)** 进行开发。我们在原作基础上进行了扩展，集成了 BERT 嵌入、开发了 Web 界面，并优化了特征工程流程。

数据来源包括：
*   **Hatebase.org**: 用于初始仇恨词典。
*   **Jeffrey Breen 的 Twitter 情感分析教程**: 用于情感词典。

---
*Developed for NLP Course Project, NLP 课程作业*
