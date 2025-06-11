# import os
# import pandas as pd

# # 1. gene_names.txt 파일 읽기 (유전자 이름만 있는 파일)
# gene_names_file = 'OncoKB_ONGene\ONGene_STRING.txt'
# gene_name_list = []

# with open(gene_names_file, 'r') as f:
#     for line in f:
#         gene_name = line.strip()
#         if gene_name:
#             gene_name_list.append(gene_name)

# # 2. multiomics 파일 읽기
# multiomics_file = 'CPDB/multiomics_features_SNV_METH_GE_CNA_filtered_CPDB.tsv'
# # multiomics_file = 'my\multiomics_features_SNV_METH_GE_CNA_filtered_STRING.tsv'
# multiomics_df = pd.read_csv(multiomics_file, sep='\t', index_col=0)

# # 3. 출력 폴더 생성
# # output_folder = '10fold_Onco_CPDB'
# output_folder = '10fold_ONGene_STRING'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 4. fold_1 ~ fold_10 반복
# for fold_num in range(1, 11):
#     fold_path = os.path.join(output_folder, f'fold_{fold_num}')
#     os.makedirs(fold_path, exist_ok=True)

#     # 5. fold 내부 test/train index 불러오기
#     # fold_data_path = f'CPDB/CPDB_10fold/fold_{fold_num}'
#     fold_data_path = f'10fold/fold_{fold_num}'
#     test_file = os.path.join(fold_data_path, 'test.txt')
#     train_file = os.path.join(fold_data_path, 'train.txt')

#     with open(test_file, 'r') as f:
#         test_indices = [int(line.strip()) for line in f]

#     with open(train_file, 'r') as f:
#         train_indices = [int(line.strip()) for line in f]

#     # 6. multiomics 인덱스 -> 유전자 이름
#     multiomics_gene_names = multiomics_df.index.tolist()

#     # 7. 마스크 초기화
#     test_mask = [0] * len(gene_name_list)
#     train_mask = [0] * len(gene_name_list)

#     # 8. 마스크 업데이트
#     for idx in test_indices:
#         if idx < len(multiomics_gene_names):
#             gene_name = multiomics_gene_names[idx]
#             if gene_name in gene_name_list:
#                 gene_idx = gene_name_list.index(gene_name)
#                 test_mask[gene_idx] = 1

#     for idx in train_indices:
#         if idx < len(multiomics_gene_names):
#             gene_name = multiomics_gene_names[idx]
#             if gene_name in gene_name_list:
#                 gene_idx = gene_name_list.index(gene_name)
#                 train_mask[gene_idx] = 1

#     # 9. 결과 저장
#     with open(os.path.join(fold_path, 'test_mask.txt'), 'w') as f:
#         f.write('\n'.join(map(str, test_mask)))

#     with open(os.path.join(fold_path, 'train_mask.txt'), 'w') as f:
#         f.write('\n'.join(map(str, train_mask)))

#     print(f"Mask files for fold {fold_num} created successfully.")


import pandas as pd

# 1. ONGene_STRING.txt 읽기 (유전자 이름 리스트)
gene_names_file = r'OncoKB_ONGene\\ONGene_STRING.txt'
with open(gene_names_file, 'r') as f:
    target_genes = [line.strip() for line in f if line.strip()]

# 2. multiomics 데이터 읽기 (인덱스 = 유전자 이름)
multiomics_file = 'my/multiomics_features_SNV_METH_GE_CNA_filtered_STRING.tsv'
multiomics_df = pd.read_csv(multiomics_file, sep='\t', index_col=0)
multiomics_genes = multiomics_df.index.tolist()

# 3. 마스크 생성: multiomics 데이터 유전자 중 target_genes 포함 여부 판단
mask = [1 if gene in target_genes else 0 for gene in multiomics_genes]

# 4. mask 배열 확인
print(f"전체 유전자 수: {len(multiomics_genes)}")
print(f"마스크된 유전자 수: {sum(mask)}")

# 5. 필요하면 저장
with open('ONGene_STRING_mask.txt', 'w') as f:
    for val in mask:
        f.write(str(val) + '\n')
