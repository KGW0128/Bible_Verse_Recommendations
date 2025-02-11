import pandas as pd

# CSV 파일 불러오기
bible_file = "./data/processed_bible.csv"
df = pd.read_csv(bible_file)

# 'tokenized'와 'filtered' 열 삭제
df.drop(columns=["tokenized", "filtered"], inplace=True, errors="ignore")

# 변경된 데이터프레임을 다시 저장
df.to_csv(bible_file, index=False)

print("✅ 'tokenized'와 'filtered' 열이 삭제되고 CSV 파일이 업데이트되었습니다!")