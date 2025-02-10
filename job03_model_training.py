import pandas as pd
from konlpy.tag import Okt
from gensim.models import Word2Vec

# 🔹 1. CSV 데이터 불러오기
file_path = "./data/merged_bible (2).csv"  # 파일 경로 변경 필요
df = pd.read_csv(file_path)

# 🔹 2. 데이터 전처리 (말씀에서 특수문자 제거)
df.dropna(inplace=True)
df['content'] = df['content'].astype(str).str.replace(r'[^가-힣\s]', '', regex=True)

# 🔹 3. 형태소 분석 (명사, 동사, 형용사만 추출)
okt = Okt()
df['tokenized'] = df['content'].apply(lambda x: [word for word, pos in okt.pos(x, stem=True) if pos in ['Noun', 'Verb', 'Adjective']])

# 🔹 4. Word2Vec 모델 학습
sentences = df['tokenized'].tolist()
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=20)

# 모델 저장 (필요 시)
word2vec_model.save('./models/word2vec_bible.model')
print("✅ Word2Vec 모델 학습 완료!")