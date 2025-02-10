import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 🔹 전처리된 데이터 불러오기
processed_file = "./data/processed_bible.csv"
df = pd.read_csv(processed_file)

# 🔹 NaN 값 및 빈 문자열 제거
df.dropna(subset=['processed'], inplace=True)  # NaN 제거
df = df[df['processed'].str.strip() != '']  # 빈 문자열 제거

# 🔹 TF-IDF 벡터화
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed'])

# 🔹 벡터화된 데이터 저장
tfidf_feature_names = vectorizer.get_feature_names_out()
tfidf_scores = dict(zip(tfidf_feature_names, np.array(tfidf_matrix.sum(axis=0)).flatten()))

# 🔹 TF-IDF 행렬 저장
import pickle
with open("./data/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

with open("./data/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ TF-IDF 벡터화 완료! `tfidf_matrix.pkl`, `tfidf_vectorizer.pkl` 저장됨.")
