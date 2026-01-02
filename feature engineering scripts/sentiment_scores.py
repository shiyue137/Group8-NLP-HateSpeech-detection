
import math
import nltk
import string
from collections import defaultdict
import math
import pandas as pd
import csv
import os
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# hate database
dict_path = r'F:\工作\上大\NLP\test2\dictionaries\hatebase_dict.csv'

# 如果 CSV 没有列名
dict1 = pd.read_csv(dict_path, encoding='ISO-8859-1', header=None)

# 给列名命名为 'word'
dict1.columns = ['word']

# 查看前 10 行
print(dict1.head())

# 获取 hate words 列
dict11 = dict1['word']
dic1 = [row.strip("', ") for row in dict11]

print(dic1[:10])

#print(dic)
# negative words lexicon
# 读取 Excel 格式的负面词典
# ============================
file_path = r'F:\工作\上大\NLP\test2\dictionaries\negative-word.csv'  # CSV 文件路径

# 读取 CSV
dict2 = pd.read_csv(file_path, encoding='utf-8-sig')

# 查看列名
print("Columns:", dict2.columns)
print(dict2.head())

# 假设实际列名是 'dic'（根据打印结果确认）
dict21 = dict2['dic']

# 去除多余空格或引号，生成列表
dic2 = [row.strip("', ") for row in dict21]

# 查看前 10 个词
print(dic2[:10])

# postive word lexicon
file_path = r'F:\工作\上大\NLP\test2\dictionaries\Postive-words.csv'

# 读取 CSV，同时去掉可能存在的 BOM
dict3 = pd.read_csv(file_path, encoding='utf-8-sig')

# 查看列名
print("Columns:", dict3.columns)
print(dict3.head())

# 假设列名是 'dic'（根据打印结果确认）
dict31 = dict3['dic']

# 去除多余空格或引号，生成列表
dic3 = [row.strip("', ") for row in dict31]

# 查看前 10 个词
print(dic3[:10])

hatedata = pd.read_csv('cleaned_tweets.csv')

tweet = hatedata['clean_tweet']
tweet1=tweet.str.split(" ")
hate = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    count = 0
    for j in tweet1[i]:
        for g in j.split(" "):
            if g in dic1:
                count+=1
        hate[i]=count

hatenor = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    l = len(tweet1[i])
    hatenor[i] = hate[i]/l

neg = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    ct = 0
    for j in tweet1[i]:
        for g in j.split(" "):
            if g in dic2:
                ct+=1
        neg[i]=ct

negnor = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    l = len(tweet1[i])
    negnor[i] = neg[i]/l

pos = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    ct1 = 0
    for j in tweet1[i]:
        for g in j.split(" "):
            if g in dic3:
                ct1+=1
        pos[i]=ct1

posnor = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    l = len(tweet1[i])
    posnor[i] = pos[i]/l

hatedata["hate"] = hate
hatedata["hatenor"] = hatenor
hatedata["neg"] = neg
hatedata["negnor"] = negnor
hatedata["pos"] = pos
hatedata["posnor"] = posnor
output_path = r'F:\工作\上大\NLP\test2\feature datasets\sentiment_scores.csv'
hatedata.to_csv(output_path, index=False)
print(f"✅ sentiment_scores.csv 已保存到: {output_path}")
print(f"✅ 包含的列: {hatedata.columns.tolist()}")
