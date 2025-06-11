import os
import pandas as pd

# 1. gene_names.txt 파일 읽기
gene_names_file = 'data/gene_names.txt'

# 원본 변환
gene_names_dict = {}

with open(gene_names_file, 'r') as f:
    gene_names_dict = {}
    gene_name_list = []
    for line in f:
        parts = line.strip().split(',')
        gene_names_dict[parts[0]] = parts[1]  # 유전자 인덱스를 키로, 유전자 이름을 값으로
        gene_name_list.append(parts[1])  # 유전자 이름 리스트로 저장

# 2. multiomics_features_SNV_METH_GE_CNA_filtered_STRING.tsv 파일 읽기
# multiomics_file = 'my/multiomics_features_SNV_METH_GE_CNA_filtered_STRING.tsv'
multiomics_file = 'CPDB\multiomics_features_SNV_METH_GE_CNA_filtered_CPDB.tsv'
multiomics_df = pd.read_csv(multiomics_file, sep='\t', index_col=0)

# 3. 10fold_new 폴더 생성 (존재하지 않으면)
# output_folder = '10fold_new' # string
output_folder = '10fold_Onco_CPDB' # CPDB
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 4. fold_1부터 fold_10까지 반복
for fold_num in range(1, 11):
    # 5. fold 폴더 생성 (각각의 fold 폴더 내에 저장)
    fold_path = os.path.join(output_folder, f'fold_{fold_num}')
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    
    # 6. test.txt와 train.txt 파일 읽기
    fold_data_path = f'CPDB/CPDB_10fold/fold_{fold_num}'
    
    test_file = os.path.join(fold_data_path, 'test.txt')
    train_file = os.path.join(fold_data_path, 'train.txt')

    with open(test_file, 'r') as f:
        test_indices = [int(line.strip()) for line in f]
    
    with open(train_file, 'r') as f:
        train_indices = [int(line.strip()) for line in f]
    
    # 7. multiomics_features_SNV_METH_GE_CNA_filtered_STRING에서 유전자 이름 인덱스 추출
    multiomics_gene_names = multiomics_df.index.tolist()
    
    # 8. 마스크 배열 생성 (0과 1로)
    test_mask = [0] * len(gene_name_list)
    train_mask = [0] * len(gene_name_list)

    # 9. test_indices와 train_indices에 대해 마스크 배열 생성 (유전자 이름 순서 기준)
    for idx in test_indices:
        if idx < len(multiomics_gene_names):
            gene_name = multiomics_gene_names[idx]
            if gene_name in gene_names_dict.values():
                # gene_name_list에서 유전자 이름 순서대로 맞춰서 마스크 업데이트
                gene_idx = gene_name_list.index(gene_name)
                test_mask[gene_idx] = 1  # TRUE (1)
    
    for idx in train_indices:
        if idx < len(multiomics_gene_names):
            gene_name = multiomics_gene_names[idx]
            if gene_name in gene_names_dict.values():
                # gene_name_list에서 유전자 이름 순서대로 맞춰서 마스크 업데이트
                gene_idx = gene_name_list.index(gene_name)
                train_mask[gene_idx] = 1  # TRUE (1)
    
    # 10. 결과 파일 저장 (test와 train 각각의 마스크 파일로 저장)
    test_mask_file = os.path.join(fold_path, f'test_mask.txt')
    train_mask_file = os.path.join(fold_path, f'train_mask.txt')
    
    # test_mask와 train_mask는 gene_name 순서 없이 마스크 값만 기록
    with open(test_mask_file, 'w') as f:
        for m in test_mask:
            f.write(f"{m}\n")

    with open(train_mask_file, 'w') as f:
        for m in train_mask:
            f.write(f"{m}\n")

    print(f"Mask files for fold {fold_num} created successfully.")
