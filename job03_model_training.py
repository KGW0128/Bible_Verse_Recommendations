
#모델학습(스케일 전)

# import pandas as pd
# from konlpy.tag import Okt
# from gensim.models import Word2Vec
# 
# # 🔹 1. CSV 데이터 불러오기
# file_path = "./data/merged_bible (2).csv"  # 파일 경로 변경 필요
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

# 🔹 1. CSV 데이터 불러오기
file_path = "./data/processed_bible.csv"  # 전처리된 성경 데이터
df = pd.read_csv(file_path)

# 🔹 2. 전처리된 'processed' 컬럼을 활용
df.dropna(subset=['processed'], inplace=True)  # 결측값 제거
df['processed'] = df['processed'].astype(str)  # 문자열 변환

# 🔹 3. 형태소 리스트로 변환
df['tokenized'] = df['processed'].apply(lambda x: x.split())

# 🔹 4. 단어 빈도 계산
word_freq = {}
for sentence in df['tokenized']:
    for word in sentence:
        word_freq[word] = word_freq.get(word, 0) + 1

# 🔹 5. 고빈도 단어 줄이기 (예: 3000번 이상 나오면 50% 확률로 제거)
THRESHOLD = 3000  # 임계값 (이 값 이상 등장하면 빈도 줄이기)
REDUCTION_PROB = 0.5  # 제거 확률 (50%)

def reduce_high_freq_words(sentence):
    return [
        word for word in sentence
        if word_freq[word] < THRESHOLD or random.random() > REDUCTION_PROB
    ]

df['filtered'] = df['tokenized'].apply(reduce_high_freq_words)

# 🔹 6. Word2Vec 모델 학습
sentences = df['filtered'].tolist()
word2vec_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=5,  # 최소 등장 횟수 설정
    workers=4,
    epochs=20,
    sample=1e-4  # 고빈도 단어 자동 샘플링
)

# 🔹 7. 모델 저장
word2vec_model.save('./models/word2vec_bible_scale.model')
print("✅ Word2Vec 모델 학습 완료! (고빈도 단어 스케일 감소 적용)")
