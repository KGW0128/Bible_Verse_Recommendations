

#백업용 코드


import pandas as pd
from konlpy.tag import Okt
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 🔹 1. CSV 데이터 불러오기
bible_file = "./data/merged_bible (2).csv"  # 성경 데이터 파일
stopwords_file = "./StopWord/stopwords.csv"  # 불용어 파일

df = pd.read_csv(bible_file)

# 🔹 2. 데이터 전처리 (결측값 제거, 특수문자 제거)
df.dropna(inplace=True)
df['content'] = df['content'].astype(str).str.replace(r'[^가-힣\s]', '', regex=True)

# 🔹 3. 불용어 목록 불러오기
stopwords_df = pd.read_csv(stopwords_file)
stopwords = set(stopwords_df['stopword'].tolist())

# 🔹 4. 형태소 분석 (명사, 동사, 형용사만 추출, 불용어 제거)
okt = Okt()
df['tokenized'] = df['content'].apply(
    lambda x: [word for word, pos in okt.pos(x, stem=True) if pos in ['Noun', 'Verb', 'Adjective']])
df['filtered'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stopwords])  # 불용어 제거
df['processed'] = df['filtered'].apply(lambda x: ' '.join(x))  # 띄어쓰기 결합

# 🔹 5. TF-IDF 벡터화
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed'])
tfidf_feature_names = vectorizer.get_feature_names_out()
tfidf_scores = dict(zip(tfidf_feature_names, np.array(tfidf_matrix.sum(axis=0)).flatten()))  # 단어별 중요도 추출

# 🔹 6. Word2Vec 모델 로드 (기존 학습된 모델 사용)
word2vec_model = Word2Vec.load('./models/word2vec_bible.model')


# 🔹 7. 기도제목에서 상위 3개 키워드 추출
def extract_keywords(prayer_text, top_n=5):
    # 1️⃣ 형태소 분석 후 명사, 동사, 형용사만 추출
    processed_prayer = [word for word, pos in okt.pos(prayer_text, stem=True) if pos in ['Noun', 'Verb', 'Adjective']]

    # 2️⃣ TF-IDF 점수가 높은 단어 중 상위 `top_n`개만 선택
    keyword_candidates = [word for word in processed_prayer if word in tfidf_scores]
    keyword_candidates = sorted(keyword_candidates, key=lambda w: tfidf_scores.get(w, 0), reverse=True)[:top_n]

    return keyword_candidates


# 🔹 8. 말씀 추천 함수 (상위 3개 키워드 기반)
def recommend_verse(prayer_text):
    keywords = extract_keywords(prayer_text, top_n=5)

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


# 🔹 9. 사용자 입력을 받아 말씀 추천 실행
while True:
    prayer_input = input("\n🙏 기도제목을 입력하세요 (종료하려면 'exit' 입력): ")
    if prayer_input.lower() == "exit":
        break
    recommend_verse(prayer_input)
