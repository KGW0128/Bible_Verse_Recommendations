import pandas as pd
import pickle
import numpy as np
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 1. 데이터 로드
df = pd.read_csv("./data/processed_bible.csv")

# 🔹 2. TF-IDF 벡터 및 행렬 로드
with open("./data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("./data/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# 🔹 3. 형태소 분석기 (Okt 사용)
okt = Okt()

# 🔹 4. TF-IDF 점수 가져오기
tfidf_feature_names = vectorizer.get_feature_names_out()
tfidf_scores = dict(zip(tfidf_feature_names, np.array(tfidf_matrix.sum(axis=0)).flatten()))  # 단어별 중요도 추출


# 🔹 5. 기도제목에서 상위 키워드 추출
def extract_keywords(prayer_text, top_n=3):
    # 형태소 분석 후 명사, 동사, 형용사만 추출
    processed_prayer = [word for word, pos in okt.pos(prayer_text, stem=True) if pos in ['Noun', 'Verb', 'Adjective']]

    # TF-IDF 점수가 높은 단어 중 상위 `top_n`개 선택
    keyword_candidates = [word for word in processed_prayer if word in tfidf_scores]
    keyword_candidates = sorted(keyword_candidates, key=lambda w: tfidf_scores.get(w, 0), reverse=True)[:top_n]

    return keyword_candidates


# 🔹 6. 성경 구절 추천 함수
def recommend_verse(prayer_text):
    keywords = extract_keywords(prayer_text, top_n=3)

    if not keywords:
        print("\n⚠️ 키워드를 추출할 수 없습니다. 다시 입력해 주세요.")
        return

    print("\n🔍 선택된 키워드:", keywords)

    # TF-IDF 유사도 계산
    query_vector = vectorizer.transform([" ".join(keywords)])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)
    top_indices = np.argsort(cosine_sim[0])[-3:][::-1]  # 유사도 높은 3개 선택

    print("\n📖 추천 성경 말씀:")
    for idx in top_indices:
        print(f"{df.iloc[idx]['book']} {df.iloc[idx]['chapter']}:{df.iloc[idx]['verse']} - {df.iloc[idx]['content']}")


# 🔹 7. 사용자 입력을 받아 말씀 추천 실행
while True:
    prayer_input = input("\n🙏 기도제목을 입력하세요 (종료하려면 'exit' 입력): ")
    if prayer_input.lower() == "exit":
        break
    recommend_verse(prayer_input)
