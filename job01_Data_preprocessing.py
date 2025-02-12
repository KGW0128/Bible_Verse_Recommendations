

#데이터 전처리 및 저장


import pandas as pd
from konlpy.tag import Okt
import collections
import numpy as np

# 🔹 데이터 파일 경로
bible_file = "./data/merged_bible.csv"  # 성경 데이터 파일
stopwords_file = "./StopWord/stopwords.csv"  # 불용어 파일
output_file = "./data/processed_bible.csv"  # 전처리된 데이터 저장 경로

# 🔹 CSV 데이터 불러오기
df = pd.read_csv(bible_file)
df.dropna(inplace=True)  # 결측값 제거

# 🔹 특수문자 제거
df['content'] = df['content'].astype(str).str.replace(r'[^가-힣\s]', '', regex=True)

# 🔹 불용어 목록 불러오기
stopwords_df = pd.read_csv(stopwords_file)
stopwords = set(stopwords_df['stopword'].tolist())

# 🔹 형태소 분석 및 전처리
okt = Okt()
df['tokenized'] = df['content'].apply(
    lambda x: [word for word, pos in okt.pos(x, stem=True) if pos in ['Noun', 'Verb', 'Adjective']]
)
df['processed'] = df['tokenized'].apply(lambda x: ' '.join([word for word in x if word not in stopwords]))

# 🔹 불필요한 열 삭제
df.drop(columns=['content', 'tokenized'], inplace=True)

# 🔹 형태소 빈도 분석
all_words = ' '.join(df['processed'].dropna()).split()
word_counts = collections.Counter(all_words)
median_value = np.median(list(word_counts.values()))

# 🔹 분석 결과 출력
print(f"🔹 형태소 총 개수: {len(word_counts)}")
print(f"🔹 중위값(중앙값) 빈도: {int(median_value)}")
print("\n🔝 가장 많이 등장한 형태소 10개:")
for word, count in word_counts.most_common(10):
    print(f"{word}: {count}회")

# 🔹 전처리된 데이터 저장
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ 데이터 전처리 완료! `{output_file}` 저장됨.")
