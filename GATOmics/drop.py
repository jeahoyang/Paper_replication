# # 'exp.txt' 파일에서 첫 번째 열과 두 번째 열을 모두 확인하고, 필터링하여 저장

# # 'exp.txt'와 'missing_gene_indices.txt' 파일 경로 설정
# exp_file = "my/PP.adj.txt"
# missing_file = "my/missing_gene_indices.txt"
# output_file = "my/filtered/PP.adj.txt"

# # missing_gene_indices.txt 파일을 읽어서 제외할 gene index들 추출
# with open(missing_file, 'r') as f:
#     missing_indices = set(int(line.strip()) for line in f.readlines())

# # exp.txt 파일을 열고 필터링된 결과를 출력 파일에 저장
# with open(exp_file, 'r') as infile, open(output_file, 'w') as outfile:
#     for line in infile:
#         cols = line.split()
#         # 첫 번째 열 또는 두 번째 열이 missing_gene_indices에 포함되지 않으면 저장
#         if int(cols[0]) not in missing_indices and int(cols[1]) not in missing_indices:
#             outfile.write(line)

# print(f"첫 번째 열과 두 번째 열을 모두 필터링하여 {output_file}에 저장했습니다.")


# import pandas as pd

# # 파일 경로 설정
# csv_file = "my/O.feat-final.csv"
# missing_file = "my/missing_gene_indices.txt"
# output_file = "my/filtered/O.feat-final.csv"

# # missing_gene_indices.txt 파일을 읽어서 제외할 gene index들 추출
# with open(missing_file, 'r') as f:
#     missing_indices = set(int(line.strip()) for line in f.readlines())

# # CSV 파일을 pandas DataFrame으로 읽기
# df = pd.read_csv(csv_file)

# # 첫 번째 열과 두 번째 열의 값이 missing_gene_indices에 포함되지 않으면 필터링
# filtered_df = df[~df.iloc[:, 0].isin(missing_indices) & ~df.iloc[:, 1].isin(missing_indices)]

# # 필터링된 DataFrame을 새 CSV 파일로 저장
# filtered_df.to_csv(output_file, index=False)

# print(f"첫 번째 열과 두 번째 열을 필터링하여 {output_file}에 저장했습니다.")


import numpy as np

# 파일 경로
label_file = "my/label_file.txt"
missing_indices_file = "my/missing_gene_indices.txt"
output_label_file = "my/filtered/label_file.txt"  # 필터링된 label을 저장할 파일 경로

# 1. label.txt 읽기
labels = np.loadtxt(label_file, dtype=int)

# 2. missing_gene_indices.txt 읽기
missing_indices = np.loadtxt(missing_indices_file, dtype=int)

# 3. label.txt의 값들 중에서 제외해야 할 인덱스를 찾기
exclude_indices = np.where(np.isin(np.arange(len(labels)), missing_indices))[0]

# 4. 제외된 인덱스를 제외한 남은 인덱스
remaining_indices = np.setdiff1d(np.arange(len(labels)), exclude_indices)

# 5. 남은 인덱스에 해당하는 값들만 필터링
filtered_labels = labels[remaining_indices]

# 6. 필터링된 결과를 새로운 파일에 저장
np.savetxt(output_label_file, filtered_labels, fmt='%d')

# 7. 결과 출력
print(f"Filtered labels saved to {output_label_file}")
