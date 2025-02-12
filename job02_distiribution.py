

# 전처리 데이터 분포도 확인



import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt

# 데이터 로드
file_path = "./data/processed_bible.csv"  # 전처리된 성경 데이터
df = pd.read_csv(file_path)

# 'processed' 컬럼에서 NaN 값 처리 후, 형태소 리스트로 변환
df.dropna(subset=['processed'], inplace=True)  # 결측값 제거
df['processed'] = df['processed'].astype(str)  # 문자열로 변환

# NaN이 아닌 문자열에 대해서만 split() 적용
df['tokenized'] = df['processed'].apply(lambda x: x.split() if isinstance(x, str) else [])

# 단어 빈도 계산
word_freq = Counter()
for sentence in df['tokenized']:
    word_freq.update(sentence)

# 단어 빈도의 분포
word_counts = list(word_freq.values())

# 분포도 시각화
plt.figure(figsize=(10,6))
plt.hist(word_counts, bins=range(1, max(word_counts)+1), edgecolor='black')
plt.title('Word Frequency Distribution')
plt.xlabel('Word Frequency')
plt.ylabel('Number of Words')
plt.show()
