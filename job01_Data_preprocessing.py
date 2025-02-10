import pandas as pd
from konlpy.tag import Okt

# 🔹 데이터 파일 경로
bible_file = "./data/merged_bible (2).csv"  # 성경 데이터 파일
stopwords_file = "./StopWord/stopwords.csv"  # 불용어 파일

# 🔹 CSV 데이터 불러오기
df = pd.read_csv(bible_file)

# 🔹 데이터 전처리 (결측값 제거, 특수문자 제거)
df.dropna(inplace=True)
df['content'] = df['content'].astype(str).str.replace(r'[^가-힣\s]', '', regex=True)

# 🔹 불용어 목록 불러오기
stopwords_df = pd.read_csv(stopwords_file)
stopwords = set(stopwords_df['stopword'].tolist())

# 🔹 형태소 분석기(Okt) 사용
okt = Okt()
df['tokenized'] = df['content'].apply(
    lambda x: [word for word, pos in okt.pos(x, stem=True) if pos in ['Noun', 'Verb', 'Adjective']]
)
df['filtered'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stopwords])  # 불용어 제거
df['processed'] = df['filtered'].apply(lambda x: ' '.join(x))  # 띄어쓰기 결합

# 🔹 전처리된 데이터 저장
df.to_csv("./data/processed_bible.csv", index=False, encoding='utf-8-sig')

print("✅ 데이터 전처리 완료! `processed_bible.csv` 저장됨.")
