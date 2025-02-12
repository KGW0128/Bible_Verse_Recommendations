
#모델학습(스케일 전)

# import pandas as pd
# from konlpy.tag import Okt
# from gensim.models import Word2Vec
#
# # 🔹 1. CSV 데이터 불러오기
# file_path = "./data/merged_bible.csv"  # 파일 경로 변경 필요
# df = pd.read_csv(file_path)
#
# # 🔹 2. 데이터 전처리 (말씀에서 특수문자 제거)
# df.dropna(inplace=True)
# df['content'] = df['content'].astype(str).str.replace(r'[^가-힣\s]', '', regex=True)
#
# # 🔹 3. 형태소 분석 (명사, 동사, 형용사만 추출)
# okt = Okt()
# df['tokenized'] = df['content'].apply(lambda x: [word for word, pos in okt.pos(x, stem=True) if pos in ['Noun', 'Verb', 'Adjective']])
#
# # 🔹 4. Word2Vec 모델 학습
# sentences = df['tokenized'].tolist()
# word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, epochs=20)
#
# # 모델 저장 (필요 시)
# word2vec_model.save('./models/word2vec_bible.model')
# print("✅ Word2Vec 모델 학습 완료!")



#----------------------------------------------------------------------

#모델학습(스케일 후)

import pandas as pd
import random
from gensim.models import Word2Vec
from collections import Counter

# 🔹 1. CSV 데이터 불러오기
file_path = "./data/processed_bible.csv"  # 전처리된 성경 데이터
df = pd.read_csv(file_path)

# 🔹 2. 전처리된 'processed' 컬럼을 활용
df.dropna(subset=['processed'], inplace=True)  # 결측값 제거
df['processed'] = df['processed'].astype(str)  # 문자열 변환

# 🔹 3. 형태소 리스트로 변환
df['tokenized'] = df['processed'].apply(lambda x: x.split())

# 🔹 4. 단어 빈도 계산
word_freq = Counter()
for sentence in df['tokenized']:
    word_freq.update(sentence)

# 🔹 5. 고빈도 단어 제한 (빈도 기준으로 700번 초과하면 700번으로 제한)
THRESHOLD = 700  # 임계값 (이 값 이상 등장하면 700번으로 제한)
def limit_high_freq_words(sentence):
    return [word if word_freq[word] <= THRESHOLD else word * (THRESHOLD // word_freq[word])
            for word in sentence]

df['filtered'] = df['tokenized'].apply(limit_high_freq_words)

# 🔹 6. Word2Vec 모델 학습
sentences = df['filtered'].tolist()
word2vec_model = Word2Vec(
    sentences,
    vector_size=200,  # 임베딩 차원 크기
    window=5,         # 문맥 단어 범위
    min_count=5,      # 최소 단어 등장 횟수
    workers=4,        # 멀티 프로세싱 활용
    epochs=50,        # 학습 에포크 수
    sg=0,             # CBOW 모델(0) vs Skip-Gram(1)
    sample=1e-4       # 고빈도 단어 자동 샘플링
)

# 🔹 7. 모델 저장
word2vec_model.save('./models/word2vec_bible_optimized.model')
print("✅ Word2Vec 모델 학습 완료! (고빈도 단어 처리 및 최적화 적용)")
