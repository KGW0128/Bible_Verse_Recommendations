import pandas as pd
import collections
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

# 🔹 한글 폰트 설정
font_path = 'malgun.ttf'  # 윈도우의 '맑은 고딕' 폰트
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)  # 설정한 폰트 적용

# 🔹 CSV 파일 불러오기
df = pd.read_csv('./data/processed_bible.csv')

# 🔹 모든 `processed` 열의 데이터를 합치기
all_words = ' '.join(df['processed'].dropna())  # NaN 값 방지

# 🔹 단어 빈도수 계산
worddict = collections.Counter(all_words.split())

# 🔹 2000회 이상 등장하는 단어 제거
THRESHOLD = 2000  # 빈도수 임계값
filtered_worddict = {word: count for word, count in worddict.items() if count <= THRESHOLD}

# 🔹 워드클라우드 생성
wordcloud_img = WordCloud(
    background_color='white',
    font_path=font_path,
    width=800,
    height=800
).generate_from_frequencies(filtered_worddict)

# 🔹 워드클라우드 출력
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off')  # 축 없애기
plt.show()
