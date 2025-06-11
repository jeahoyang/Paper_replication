import pandas as pd

# 파일 경로 설정
gene_names_file = "data/gene_names.txt"
multiomics_file = "my/multiomics_features_SNV_METH_GE_CNA_filtered_STRING.tsv"

# 유전자 이름 목록 읽기
with open(gene_names_file, 'r') as f:
    gene_names = [line.split(',')[1].strip() for line in f.readlines()]

# multiomics 데이터프레임 읽기
multiomics_data = pd.read_csv(multiomics_file, sep="\t")

# 첫 번째 열이 유전자 이름이므로 해당 열만 추출
multiomics_genes = multiomics_data.iloc[:, 0].values
print(multiomics_genes)

# gene_names에 있는 유전자 이름이 multiomics 데이터에 없는 인덱스 찾기
missing_gene_indices = [index for index, gene in enumerate(gene_names) if gene not in multiomics_genes]

# 결과 출력
print(f"multiomics 데이터에 없는 유전자 이름의 인덱스: {missing_gene_indices}")

# 인덱스를 파일에 저장
with open("missing_gene_indices.txt", 'w') as f:
    for idx in missing_gene_indices:
        f.write(f"{idx}\n")

print("multiomics 데이터에 없는 유전자 인덱스가 'missing_gene_indices.txt'에 저장되었습니다.")
