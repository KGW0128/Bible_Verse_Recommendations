
성경을 이용하여 성경을 학습시켜 기도제목을 적으면 추천 말씀을 주는 AI프로그램


# 📖 Bible Verse Recommendation App

이 프로젝트는 사용자가 입력한 **기도 제목**을 분석하여, 관련된 성경 구절을 추천하는 **성경 구절 추천 시스템**입니다.

## ✨ 주요 기능
- **기도 제목 입력:** 사용자가 원하는 기도 제목을 입력하면
- **형태소 분석 & 키워드 추출:** 자연어 처리를 이용해 핵심 단어를 추출
- **TF-IDF + Word2Vec 유사도 분석:** 단어의 문맥적 의미까지 고려한 성경 구절 추천
- **배경 이미지 & 스타일 적용된 GUI:** PyQt5 기반의 간단한 GUI 제공

---

## 🛠 코드의 주요 흐름

1. **사용자가 기도 제목을 입력**하면 → `btn_slot()` 함수 실행
2. **형태소 분석 (Okt) + 불용어 제거**
   - `extract_keywords()` 함수에서 **형태소 분석(Okt)**을 사용하여 **명사, 동사, 형용사**만 추출
   - 불용어(stopwords)를 제거
   - 추출된 단어 중 TF-IDF 벡터에 존재하는 단어만 필터링
   - TF-IDF 점수가 높은 단어들 **(상위 5개 선택)**
3. **TF-IDF 코사인 유사도 계산**
   - `vectorizer.transform()`으로 기도 제목을 벡터로 변환
   - `cosine_similarity()`를 사용해 성경 구절 벡터들과 비교
4. **Word2Vec 유사도 계산**
   - `get_word2vec_similarity()` 함수에서 **기도 제목의 키워드**와 **각 성경 구절 단어들** 간 **Word2Vec 유사도** 계산
   - (모델이 학습한 단어들만 비교 가능)
5. **최종 점수 계산**
   - TF-IDF 코사인 유사도 **(70%)** + Word2Vec 유사도 **(30%)** 가중치 적용
   - 상위 3개 구절을 추천하여 UI에 표시

---

## 🖥 실행 방법
### 1️⃣ 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2️⃣ 실행
```bash
python app.py
```

---

## 📂 프로젝트 구조
```
📂 BibleVerseRecommendation
│── 📂 data
│   ├── processed_bible.csv   # 성경 데이터셋
│   ├── stopwords.csv         # 불용어 리스트
│   ├── bible_search_keywords.csv # 검색 키워드
│── 📂 models
│   ├── word2vec_bible_scale.model  # Word2Vec 모델
│   ├── tfidf_vectorizer.pkl        # TF-IDF 벡터 변환기
│   ├── tfidf_matrix.pkl            # TF-IDF 행렬
│── 📂 images
│   ├── img_2.jpg     # GUI 배경 이미지
│── ui.ui            # PyQt5 UI 파일
│── app.py           # 메인 코드
│── README.md        # 프로젝트 설명 파일
│── requirements.txt # 설치해야 할 패키지 목록
```

---

## 📌 추가 개발 방향
✅ 검색 속도 최적화
✅ Word2Vec 가중치 조절 옵션 추가
✅ 특정 성경책에서만 검색하는 기능 추가

---

### 🤝 기여 방법
1. 프로젝트를 포크합니다.
2. 새로운 기능을 개발하고, 테스트합니다.
3. PR(Pull Request)을 생성합니다.

🙌 많은 관심과 기여 부탁드립니다!

