import os
import pandas as pd

# 1. merged_result.csv 로드
merged_result_path = "merged_result.csv"  # 파일 경로 지정
df = pd.read_csv(merged_result_path)  # CSV 읽기

# 2. Index -> GeneID 매핑 딕셔너리 생성
index_to_geneid = dict(zip(df["index"], df["GeneID"]))  # index 값을 key, GeneID를 value로 매핑

# 3. 10개의 fold 폴더를 순회하며 txt 파일 변환
base_folder = "STRING_10foldTEST"  # 기본 폴더 경로

for fold in range(1, 11):  # fold_1 ~ fold_10
    fold_path = os.path.join(base_folder, f"fold_{fold}")  # fold 폴더 경로
    for file_name in ["train.txt", "test.txt", "valid.txt"]:  # 변환할 파일 목록
        file_path = os.path.join(fold_path, file_name)  # txt 파일 경로

        # 3-1. 파일이 존재하는지 확인
        if not os.path.exists(file_path):
            print(f"파일 없음: {file_path}")
            continue

        # 3-2. txt 파일 읽어서 index 값을 GeneID로 변환
        with open(file_path, "r") as f:
            indices = [int(line.strip()) for line in f.readlines()]  # 줄 단위로 읽어서 숫자로 변환

        # 3-3. index 값을 GeneID로 변환
        gene_ids = [index_to_geneid[idx] if idx in index_to_geneid else f"Unknown_{idx}" for idx in indices]

        # 3-4. 변환된 결과를 다시 저장 (덮어쓰기)
        with open(file_path, "w") as f:
            f.write("\n".join(map(str, gene_ids)))

        print(f"변환 완료: {file_path}")
