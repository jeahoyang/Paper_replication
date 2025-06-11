import os
import pandas as pd

# 1. gene_names.txt 읽기
gene_names_file = 'data/gene_names.txt'
gene_names_dict = {}
gene_name_list = []

with open(gene_names_file, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        gene_names_dict[parts[0]] = parts[1]  # 유전자 인덱스를 키로, 유전자 이름을 값으로
        gene_name_list.append(parts[1])  # 유전자 이름 리스트로 저장

# 2. multiomics_features_SNV_METH_GE_CNA_filtered_STRING.tsv 파일 읽기
multiomics_file = 'my/multiomics_features_SNV_METH_GE_CNA_filtered_STRING.tsv'
multiomics_df = pd.read_csv(multiomics_file, sep='\t', index_col=0)

# 3. multiomics의 gene 이름을 gene_names.txt 기준으로 변환
multiomics_gene_names = multiomics_df.index.tolist()
mapped_indices = []

gene_name_to_index = {gene: i for i, gene in enumerate(gene_name_list)}

for gene in multiomics_gene_names:
    if gene in gene_name_to_index:
        mapped_indices.append(gene_name_to_index[gene])
    else:
        mapped_indices.append(-1)  # 매칭되지 않는 경우 -1 저장

# 4. 변환된 인덱스를 파일로 저장
mapped_indices_file = 'mapped_gene_indices.txt'
with open(mapped_indices_file, 'w') as f:
    for idx in mapped_indices:
        f.write(f"{idx}\n")

print(f"Mapped indices saved to {mapped_indices_file}")

# 5. specific-cancer 폴더 내 파일 목록
specific_cancer_folder = 'specific-cancer'
new_output_folder = 'specific-cancer-updated'
if not os.path.exists(new_output_folder):
    os.makedirs(new_output_folder)

files_to_update = [
    'pan-neg.txt', 'pos-blca.txt', 'pos-brca.txt', 'pos-cesc.txt', 'pos-coad.txt',
    'pos-esca.txt', 'pos-hnsc.txt', 'pos-kirc.txt', 'pos-kirp.txt', 'pos-lihc.txt',
    'pos-luad.txt', 'pos-lusc.txt', 'pos-prad.txt', 'pos-ucec.txt', 'pos-stad.txt',
    'pos-thca.txt'
]

# 6. 각 파일의 기존 인덱스를 변환하여 저장
for filename in files_to_update:
    input_file = os.path.join(specific_cancer_folder, filename)
    output_file = os.path.join(new_output_folder, filename)  # 새로운 폴더에 저장
    
    updated_indices = []
    
    with open(input_file, 'r') as f:
        for line in f:
            original_index = int(line.strip())  # multiomics 기준 인덱스
            if 0 <= original_index < len(mapped_indices):
                new_index = mapped_indices[original_index]  # 변환된 gene_names 기준 인덱스
                updated_indices.append(str(new_index))
    
    # 변환된 인덱스를 새로운 폴더 내 파일로 저장
    with open(output_file, 'w') as f:
        f.write('\n'.join(updated_indices) + '\n')
    
    print(f'Updated: {filename} -> {output_file}')

# 7. label 파일 업데이트
label_files_to_update = [
    'label_file-P-blca.txt', 'label_file-P-brca.txt', 'label_file-P-cesc.txt', 'label_file-P-coad.txt',
    'label_file-P-esca.txt', 'label_file-P-hnsc.txt', 'label_file-P-kirc.txt', 'label_file-P-kirp.txt',
    'label_file-P-lihc.txt', 'label_file-P-luad.txt', 'label_file-P-lusc.txt', 'label_file-P-prad.txt',
    'label_file-P-stad.txt', 'label_file-P-thca.txt', 'label_file-P-ucec.txt'
]

new_label_output_folder = 'specific-cancer-label-updated'
if not os.path.exists(new_label_output_folder):
    os.makedirs(new_label_output_folder)

for filename in label_files_to_update:
    input_file = os.path.join(specific_cancer_folder, filename)
    output_file = os.path.join(new_label_output_folder, filename)
    
    label_indices = []
    
    with open(input_file, 'r') as f:
        for idx, line in enumerate(f):
            if int(line.strip()) == 1:
                label_indices.append(idx)  # label이 1인 인덱스만 저장
    
    # 기존 multiomics 인덱스를 gene_name 기준으로 변환
    updated_labels = ["0"] * len(gene_name_list)
    for idx in label_indices:
        if 0 <= idx < len(mapped_indices) and mapped_indices[idx] != -1:
            updated_labels[mapped_indices[idx]] = "1"
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(updated_labels) + '\n')
    
    print(f'Updated: {filename} -> {output_file}')