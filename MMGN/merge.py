import pandas as pd

# 715true.txt 불러오기 (탭 구분)
df_715true = pd.read_csv('715true.txt', sep='\t')

# other.csv 불러오기 (쉼표 구분)
df_other = pd.read_csv('Homo_sapiens.csv')

# 'gene' 기준으로 병합 (inner join)
merged_df = df_other.merge(df_715true, on='gene', how='inner')

# 결과 확인
print(merged_df)

# 병합된 데이터를 새로운 CSV로 저장
merged_df.to_csv('715true.csv', index=False)
