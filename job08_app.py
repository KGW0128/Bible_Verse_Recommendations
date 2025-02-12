

# 🔹 코드의 주요 흐름

# 1. 사용자가 기도 제목을 입력하면 → btn_slot() 함수 실행

# 2. 형태소 분석 (Okt) + 불용어 제거
#    - extract_keywords() 함수에서 형태소 분석(Okt)을 사용해서 명사, 동사, 형용사만 추출
#    - 불용어(stopwords)를 제거
#    - 추출된 단어 중 TF-IDF 벡터에 존재하는 단어만 필터링

# 3. TF-IDF 점수가 높은 단어들 (상위 5개 선택)

# 4. TF-IDF 코사인 유사도 계산
#    - vectorizer.transform()으로 기도 제목을 벡터로 변환
#    - cosine_similarity()를 사용해 성경 구절 벡터들과 비교

# 5. Word2Vec 유사도 계산
#    - get_word2vec_similarity() 함수에서 기도 제목의 키워드와 각 성경 구절 단어들 간 Word2Vec 유사도 계산
#    - (모델이 학습한 단어들만 비교)

# 6. 최종 점수 계산
#    - TF-IDF 코사인 유사도 (70%) + Word2Vec 유사도 (30%) 가중치 적용
#    - 상위 3개 구절을 추천



import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5 import uic
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
from gensim.models import Word2Vec
from scipy.io import mmread  # 희소 행렬 로드
import pickle  # 객체 직렬화

# ✅ UI 파일 로드
from_window = uic.loadUiType('./Word_recommendation.ui')[0]


# 🔹 BibleApp 클래스 정의
class BibleApp(QWidget, from_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


        self.setFixedSize(550, 700)  # 창 크기를 고정


        # 🔹 배경 이미지 설정
        self.set_background("images/img_2.jpg")

        # 🔹 성경 데이터 불러오기
        self.df = pd.read_csv("./data/merged_bible.csv")  # content 컬럼이 있는 파일
        self.processed_df = pd.read_csv("./data/processed_bible.csv")  # processed 컬럼이 있는 파일

        # 🔹 TF-IDF 희소 행렬(.mtx 파일) 로드
        self.tfidf_matrix = mmread("./data/tfidf_matrix.mtx").toarray()

        # 🔹 TF-IDF 벡터화기 객체 로드 (Vectorizer)
        with open("./data/tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

        # 🔹 Word2Vec 모델 로드
        self.word2vec_model = Word2Vec.load("./models/word2vec_bible_optimized.model")

        # 🔹 형태소 분석기 및 불용어 목록
        self.okt = Okt()
        stopwords_df = pd.read_csv("./StopWord/stopwords.csv")
        self.stopwords = set(stopwords_df["stopword"].tolist())

        # 🔹 데이터 크기 맞추기
        self.df = self.df.iloc[:self.tfidf_matrix.shape[0]]

        # 🔹 UI 연결
        self.btn_recommend.clicked.connect(self.btn_slot)  # 버튼 클릭 시 실행
        self.le_keyword.returnPressed.connect(self.btn_slot)  # 엔터 키 입력 시 실행

        # 🔹 시작 문자 출력
        self.lbl_input_phrase.setText("🙏 기도 제목을 적어주세요.")

        # 🔹 QLabel 자동 줄바꿈 설정
        self.lbl_recommadation.setWordWrap(True)

        # 🔹 버튼 스타일 적용 ✅
        self.btn_recommend.setStyleSheet("""
            QPushButton {
                background-color: #D5D5D5;
                border: 2px solid #9F9F9F;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #B1B1B1;
            }
            QPushButton:pressed {
                background-color: #7B7B7B;
            }
        """)

        # 🔹 입력창 스타일 적용 ✅
        self.le_keyword.setStyleSheet("""
            QLineEdit {
                background-color: #D5D5D5;
                border: 2px solid #9F9F9F;
                font-size: 16px;
                padding: 8px;
                border-radius: 8px;
            }
        """)

        # 🔹 콤보박스 초기화 ✅
        self.load_keywords()
        self.cb_title.currentIndexChanged.connect(self.fill_keyword)

    def load_keywords(self):
        """ 🔹 'bible_search_keywords.csv'에서 키워드를 로드하여 콤보박스에 추가 """
        try:
            keywords_df = pd.read_csv("./data/bible_search_keywords.csv", header=None)
            keywords = keywords_df[0].dropna().tolist()[1:]  # NaN 값 제거 후 리스트 변환(첫 줄 제외)
            self.cb_title.addItems([""] + keywords)  # 기본값 추가
        except Exception as e:
            print(f"⚠️ 키워드 로드 실패: {e}")

    def fill_keyword(self):
        """ 🔹 콤보박스에서 선택한 키워드를 입력창에 넣고 자동 검색 실행 """
        selected_keyword = self.cb_title.currentText()
        if selected_keyword != "":
            self.le_keyword.setText(selected_keyword)
            self.btn_slot()

    def set_background(self, image_path):
        """ 🔹 QLabel을 이용한 배경 이미지 설정 """
        self.bg_label = QLabel(self)
        self.bg_pixmap = QPixmap(image_path)
        self.bg_label.setScaledContents(True)
        self.bg_label.lower()
        self.update_background_size()

    def resizeEvent(self, event):
        """ 🔹 윈도우 크기 변경 시 배경 이미지 크기 조정 """
        self.update_background_size()
        super().resizeEvent(event)

    def update_background_size(self):
        """ 🔹 배경 이미지의 크기 조정 """
        self.bg_label.setGeometry(0, 0, self.width(), self.height())
        self.bg_label.setPixmap(self.bg_pixmap.scaled(self.width(), self.height()))

    def btn_slot(self):
        """ 🔹 버튼 클릭 시 실행될 함수 """
        self.lbl_recommadation.setText("⏳ 말씀을 찾고 있습니다...")
        QApplication.processEvents()

        prayer_text = self.le_keyword.text().strip()
        if not prayer_text:
            self.lbl_recommadation.setText("⚠️ 기도 제목을 입력하세요.")
            return

        recommendations = self.recommend_verse(prayer_text, top_n=3)
        self.lbl_recommadation.setText("\n\n".join(recommendations))

    def extract_keywords(self, prayer_text, top_n=10):
        """ 🔹 기도 제목에서 키워드 추출 함수 """
        processed_prayer = [
            word for word, pos in self.okt.pos(prayer_text, stem=True)
            if pos in ["Noun", "Verb", "Adjective"]
        ]
        filtered_prayer = [word for word in processed_prayer if word not in self.stopwords]
        keyword_candidates = [
            word for word in filtered_prayer if word in self.vectorizer.get_feature_names_out()
        ]
        keyword_candidates = sorted(
            keyword_candidates,
            key=lambda w: self.tfidf_matrix[:, self.vectorizer.vocabulary_.get(w, 0)].sum(),
            reverse=True
        )[:top_n]
        return keyword_candidates

    def get_word2vec_similarity(self, keywords, verse_words):
        """ 🔹 Word2Vec 유사도 계산 """
        similarities = [
            self.word2vec_model.wv.similarity(k, w)
            for k in keywords for w in verse_words
            if k in self.word2vec_model.wv and w in self.word2vec_model.wv
        ]
        return np.mean(similarities) if similarities else 0

    def recommend_verse(self, prayer_text, top_n=3):
        """ 🔹 성경 구절 추천 함수 """
        keywords = self.extract_keywords(prayer_text, top_n=10)
        if not keywords:
            return ["⚠️ 키워드를 추출할 수 없습니다. 다시 입력해 주세요."]

        # TF-IDF 벡터화한 쿼리 생성
        query_vector = self.vectorizer.transform([" ".join(keywords)])
        cosine_sim = cosine_similarity(query_vector, self.tfidf_matrix)

        # Word2Vec 유사도 계산
        self.df["word2vec_sim"] = self.processed_df["processed"].apply(
            lambda x: self.get_word2vec_similarity(keywords, str(x).split()) if isinstance(x, str) else 0
        )

        # TF-IDF (70%) + Word2Vec (30%) 결합
        final_scores = (cosine_sim[0] * 0.7) + (self.df["word2vec_sim"].values * 0.3)
        top_indices = np.argsort(final_scores)[-top_n:][::-1]

        return [
            f"📖 {self.df.iloc[idx]['book']} {self.df.iloc[idx]['chapter']}:{self.df.iloc[idx]['verse']} - {self.df.iloc[idx]['content']}"
            for idx in top_indices
        ]


# 🔹 실행 코드
def main():
    app = QApplication(sys.argv)
    ex = BibleApp()
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
