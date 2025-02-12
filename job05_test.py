import pandas as pd
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.io import mmread

# 🔹 1. 성경 데이터 불러오기 (전처리된 CSV 파일)
bible_file = "./data/processed_bible.csv"
merged_bible_file = "./data/merged_bible.csv"

# processed_bible.csv에서 'processed' 열을 불러오기
df_processed = pd.read_csv(bible_file)

# merged_bible.csv에서 'content' 열을 불러오기
df_merged = pd.read_csv(merged_bible_file)

# 🔹 2. TF-IDF 모델 불러오기
#   - TF-IDF: 단어의 중요도를 평가하는 방법 (단어 빈도 기반)
#   - vectorizer: 텍스트를 벡터로 변환하는 객체
#   - tfidf_matrix: 성경 구절 전체를 벡터화한 결과
with open("./data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 🔹 3. .mtx 형식의 TF-IDF 행렬 불러오기
tfidf_matrix = mmread("./data/tfidf_matrix.mtx").tocsc()  # .mtx 형식으로 불러오기

# 🔹 4. Word2Vec 모델 불러오기
#   - 단어 간 유사도를 계산하기 위한 모델
word2vec_model = Word2Vec.load("./models/word2vec_bible_optimized.model")

# 🔹 5. 형태소 분석기 및 불용어 목록 설정
#   - Okt: 한국어 형태소 분석기 (명사, 동사, 형용사만 추출)
#   - 불용어(stopwords): 의미 없는 단어 제거를 위해 사용
okt = Okt()
stopwords_df = pd.read_csv("./StopWord/stopwords.csv")
stopwords = set(stopwords_df["stopword"].tolist())

# 🔹 6. 데이터 크기 맞추기
#   - TF-IDF 벡터화 과정에서 일부 행이 삭제될 수 있음
#   - df_processed의 크기를 tfidf_matrix 크기에 맞춰 조정
df_processed = df_processed.iloc[:tfidf_matrix.shape[0]]

# 🔹 7. 기도 제목에서 키워드 추출하는 함수
def extract_keywords(prayer_text, top_n=10):
    """
    사용자가 입력한 기도 제목에서 중요한 키워드를 추출하는 함수.

    1. 형태소 분석을 통해 명사, 동사, 형용사만 추출
    2. 불용어(stopwords) 제거
    3. TF-IDF 값이 높은 상위 top_n개의 키워드를 선택

    :param prayer_text: 사용자가 입력한 기도 제목 (문장)
    :param top_n: 추출할 키워드 개수 (기본값 5개)
    :return: 최종적으로 선택된 키워드 리스트
    """
    # 1️⃣ 형태소 분석 및 품사 태깅 (명사, 동사, 형용사만 선택)
    processed_prayer = [word for word, pos in okt.pos(prayer_text, stem=True) if pos in ["Noun", "Verb", "Adjective"]]

    # 2️⃣ 불용어 제거
    filtered_prayer = [word for word in processed_prayer if word not in stopwords]

    # 3️⃣ TF-IDF 단어 목록에서 기도 제목과 겹치는 단어만 선택
    keyword_candidates = [word for word in filtered_prayer if word in vectorizer.get_feature_names_out()]

    # 4️⃣ TF-IDF 값이 높은 단어 순으로 정렬 후 상위 N개 선택
    keyword_candidates = sorted(keyword_candidates,
                                key=lambda w: tfidf_matrix[:, vectorizer.vocabulary_.get(w, 0)].sum(),
                                reverse=True)[:top_n]

    return keyword_candidates


# 🔹 8. Word2Vec 기반 유사도 계산 함수
def get_word2vec_similarity(keywords, verse_words):
    """
    성경 구절과 기도 제목 키워드 간의 유사도를 계산하는 함수.

    1. 기도 제목의 각 키워드와 성경 구절의 각 단어 간 유사도 계산
    2. 모든 유사도의 평균값을 반환

    :param keywords: 기도 제목에서 추출된 키워드 리스트
    :param verse_words: 성경 구절을 단어 단위로 나눈 리스트
    :return: 평균 유사도 값 (0~1)
    """
    similarities = []

    for keyword in keywords:
        for word in verse_words:
            # 단어가 Word2Vec 모델에 있을 경우 유사도 계산
            if keyword in word2vec_model.wv and word in word2vec_model.wv:
                similarities.append(word2vec_model.wv.similarity(keyword, word))

    # 평균 유사도 값 반환 (유사도 값이 없으면 0 반환)
    return np.mean(similarities) if similarities else 0


# 🔹 9. 성경 구절 추천 함수
def recommend_verse(prayer_text):
    """
    기도 제목을 입력받아 관련된 성경 구절을 추천하는 함수.
    """
    keywords = extract_keywords(prayer_text, top_n=10)

    if not keywords:
        print("\n⚠️ 키워드를 추출할 수 없습니다. 다시 입력해 주세요.")
        return

    print("\n🔍 선택된 키워드:", keywords)

    # 1️⃣ TF-IDF 유사도 계산 (기도 제목과 성경 구절 간의 유사도)
    query_vector = vectorizer.transform([" ".join(keywords)])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)

    # 2️⃣ Word2Vec 유사도 계산 (단어 의미 기반 유사도)
    df_processed["word2vec_sim"] = df_processed["processed"].apply(
        lambda x: get_word2vec_similarity(keywords, str(x).split()) if isinstance(x, str) else 0)

    # 3️⃣ TF-IDF 유사도(70%) + Word2Vec 유사도(30%) 결합
    final_scores = (cosine_sim[0] * 0.7) + (df_processed["word2vec_sim"].values * 0.3)

    # 4️⃣ 유사도가 높은 상위 3개 성경 구절 찾기
    top_indices = np.argsort(final_scores)[-3:][::-1]

    # 5️⃣ 추천 성경 구절 출력
    print("\n📖 추천 성경 말씀:")
    for idx in top_indices:
        print(f"{df_merged.iloc[idx]['book']} {df_merged.iloc[idx]['chapter']}:{df_merged.iloc[idx]['verse']} - {df_merged.iloc[idx]['content']}")


# 🔹 10. 사용자 입력을 받아 성경 구절 추천 실행
while True:
    prayer_input = input("\n🙏 기도제목을 입력하세요 (종료하려면 'exit' 입력): ")
    if prayer_input.lower() == "exit":
        break
    recommend_verse(prayer_input)




# 🔹 TF-IDF와 Word2Vec 유사도 결합 (70% : 30% 비율)
#
# TF-IDF:
# - 텍스트에서 단어의 중요도를 평가하는 방식으로, 단어가 문서 내에서 얼마나 중요한지 기반으로 유사도를 계산합니다.
# - 성경 구절에서 단어 빈도와 문서 내 단어의 분포를 고려하므로, 기도 제목에 포함된 단어가 성경 구절과 얼마나 일치하는지 직관적으로 계산할 수 있습니다.
# - 강점: 특정 단어의 중요도를 잘 반영하고, 구체적인 단어 일치를 기반으로 유사도를 계산합니다.
# - 단점: 단어의 의미나 맥락을 반영하는 데 한계가 있습니다.
#
# Word2Vec:
# - 단어의 의미적 유사성을 기반으로 계산합니다. 단어가 벡터화되어 의미가 비슷한 단어들끼리 가까운 벡터 공간에 위치합니다.
# - 기도 제목의 키워드가 성경 구절의 단어와 의미적으로 얼마나 유사한지를 평가하는 데 강점을 보입니다.
# - 강점: 의미적 유사성을 잘 반영하고, 단어의 맥락을 고려한 유사도 계산이 가능합니다.
# - 단점: 단어 간 의미적 유사성은 있을지라도, 문서 내의 중요도나 빈도 차이는 반영되지 않습니다.
#
# 유사도 결합 비율:
# - 70% TF-IDF: 정확한 단어 일치를 강조하며 성경 구절에서 중요한 단어가 어떻게 배치되는지 평가합니다. 정확히 일치하는 단어들을 강조하기 위해 TF-IDF 비율을 높게 설정합니다.
# - 30% Word2Vec: 단어의 의미적 유사성을 평가하며, 의미적으로 유사한 단어들 간의 유사도를 반영합니다. 의미적인 차이를 반영하기 위해 Word2Vec 비율을 설정합니다.

