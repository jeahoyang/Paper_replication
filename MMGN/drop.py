import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('STRING_ppi_edgelist_mapped.csv')

# 필요한 열만 선택
df = df[['partner1','partner2']]

# 결과 저장
df.to_csv('STRING_ppi_edgelist_mapped.csv', index=False)

# 출력 확인
print(df.head())
