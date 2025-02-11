
#키워스 생성
#콤보 박스에 넣을 용도


import pandas as pd

# 🔹 사람들이 성경을 검색할 때 많이 쓰는 키워드 30개 선정
keywords = [
    "기도", "말씀", "위로", "축복", "사랑", "고난", "소망", "용서", "인내", "감사",
    "회개", "구원", "예배", "찬양", "십자가", "부활", "은혜", "믿음", "성령", "평안",
    "치유","건강", "지혜", "인도", "심판", "용기", "순종", "회복", "기쁨", "사명", "진리"
]

# 🔹 DataFrame 생성
df_keywords = pd.DataFrame({"keyword": keywords})

# 🔹 CSV 파일로 저장
df_keywords.to_csv("./data/bible_search_keywords.csv", index=False, encoding="utf-8-sig")

print("✅ 'bible_search_keywords.csv' 파일이 생성되었습니다!")
