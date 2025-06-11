import os

# 1) fold prediction gene 추출
prediction_folder = "prediction_CPDB"
all_genes = set()

for i in range(1, 11):
    file_path = os.path.join(prediction_folder, f"predictions_fold_{i}.txt")
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # 첫 줄 header 스킵, 각 줄에서 첫번째 컬럼 gene 추출
        for line in lines[1:]:
            gene = line.strip().split('\t')[0]
            all_genes.add(gene)

print(f"전체 예측된 gene 수: {len(all_genes)}")

# 2) OncoKB gene 리스트 읽기
onco_file = "OncoKB_ONGene/OncoKB_CPDB.txt"
with open(onco_file, 'r') as f:
    onco_genes = set(line.strip() for line in f if line.strip())

print(f"OncoKB gene 수: {len(onco_genes)}")

# 3) 겹치는 gene 찾기
overlap_genes = all_genes.intersection(onco_genes)
print(f"겹치는 gene 수: {len(overlap_genes)}")
print("겹치는 gene 목록:")
for gene in sorted(overlap_genes):
    print(gene)
