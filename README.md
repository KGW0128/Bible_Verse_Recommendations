
성경을 학습시켜 기도제목을 적으면 추천 말씀을 주는 AI프로그램(+ui)<br>
TF-IDF와 Word2Vec.model을 결합하여 제작

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
   - TF-IDF 점수가 높은 단어들 **(상위 10개 선택)**
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
python job08_app.py
```

---

## 📂 프로젝트 구조
```
📂 BibleVerseRecommendation
│── 📂 data
│   ├── bible_search_keywords.csv         # 추천 키워드 (수동입력한 데이터)
│   ├── merged_bible.csv                  # 성경 데이터셋
│   ├── processed_bible.csv               # 전처리한 데이터셋
│   ├── tfidf_matrix.mtx                  # TFIDF의 행렬 .mtx파일
│   ├── tfidf_matrix.pkl                  # TFIDF의 통합 .pkl파일
│   ├── tfidf_vectorizer.pkl              # TFIDF의 벡터라이저 파일 (.mtx와 같이 사용)
│
│── 📂 images
│   ├── bible_image.png                   # GUI 실행화면 이미지
│   ├── img_1/2/3.jpg                     # GUI 배경 이미지 1/2/3
│
│── 📂 models
│   ├── word2vec_bible.model              # Word2Vec 초기모델
│   ├── word2vec_bible_optimized.model    # Word2Vec 스케일 후 수치조정한 3번째 모델
│   ├── word2vec_bible_scale.model        # Word2Vec 데이터 스케일 2번째 모델
│
│── 📂 StopWord
│   ├── stopwords.csv                     # 불용어 처리용 파일
│
│
│── job00_backup_code.py                  # 백업용 파일 (job_08_app.py)(TFIDF .pkl 버전)
│── job01_Data_preprocessing.py           # 성경 데이터 전처리
│── job02_distiribution.py                # 전처리 데이터 분포도 확인
│── job03_vectorize.py                    # 전처리된 데이터를 tfidf로 백터화 및 행렬저장
│── job04_model_training.py               # 모델 학습 (스케일 적용)
│── job05_test.py                         # ui없이 테스트용 실행 (text 출력)
│── job06_wordCloud.py                    # Word Cloud 확인
│── job07_morpheme_extraction.py          # 추천 키워드 생성 (수동입력)
│── job08_app.py                          # UI실행 메인 코드
│
│
│── malgun.ttf                            # Word Cloud 출력용 폰트
│── README.md                             # 프로젝트 설명 파일
│── requirements.txt                      # 설치해야 할 패키지 목록
│── Word_recommendation.ui                # PyQt5 UI 파일


```

---

## 📌 추가 개발 방향
✅ 검색 속도 최적화
✅ 완성도를 위한 전체 수치조절
✅ 모바일 앱 제작

---

### 🤝 기여 방법
1. 프로젝트를 포크합니다.
2. 새로운 기능을 개발하고, 테스트합니다.
3. PR(Pull Request)을 생성합니다.

🙌 많은 관심과 기여 부탁드립니다!

---

### 추가 설명

🔹 TF-IDF와 Word2Vec 유사도 결합 (70% : 30% 비율)

TF-IDF:
- 텍스트에서 단어의 중요도를 평가하는 방식으로, 단어가 문서 내에서 얼마나 중요한지 기반으로 유사도를 계산합니다.
- 성경 구절에서 단어 빈도와 문서 내 단어의 분포를 고려하므로, 기도 제목에 포함된 단어가 성경 구절과 얼마나 일치하는지 직관적으로 계산할 수 있습니다.
- 강점: 특정 단어의 중요도를 잘 반영하고, 구체적인 단어 일치를 기반으로 유사도를 계산합니다.
- 단점: 단어의 의미나 맥락을 반영하는 데 한계가 있습니다.

Word2Vec:
- 단어의 의미적 유사성을 기반으로 계산합니다. 단어가 벡터화되어 의미가 비슷한 단어들끼리 가까운 벡터 공간에 위치합니다.
- 기도 제목의 키워드가 성경 구절의 단어와 의미적으로 얼마나 유사한지를 평가하는 데 강점을 보입니다.
- 강점: 의미적 유사성을 잘 반영하고, 단어의 맥락을 고려한 유사도 계산이 가능합니다.
- 단점: 단어 간 의미적 유사성은 있을지라도, 문서 내의 중요도나 빈도 차이는 반영되지 않습니다.

유사도 결합 비율:
- 70% TF-IDF: 정확한 단어 일치를 강조하며 성경 구절에서 중요한 단어가 어떻게 배치되는지 평가합니다. 정확히 일치하는 단어들을 강조하기 위해 TF-IDF 비율을 높게 설정합니다.
- 30% Word2Vec: 단어의 의미적 유사성을 평가하며, 의미적으로 유사한 단어들 간의 유사도를 반영합니다. 의미적인 차이를 반영하기 위해 Word2Vec 비율을 설정합니다.


