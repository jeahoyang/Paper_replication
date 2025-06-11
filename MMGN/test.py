import pandas as pd

# 파일 경로 설정
ppi_file = "processed_dataset/six_net/PPI_for_pyG.txt"
mapping_file = "mapping_dict_EIDtoIndex.txt"
output_file = "processed_dataset/six_net/PPI_filtered.txt"  # 새로운 PPI 파일 저장 경로

# PPI 파일에서 첫 번째 열과 두 번째 열의 숫자를 집합(set)으로 저장
ppi_numbers = set()
ppi_lines = []  # 필터링 후 저장할 줄

with open(ppi_file, "r") as f:
    for line in f:
        nums = line.strip().split()
        if len(nums) == 2:
            ppi_numbers.add(int(nums[0]))
            ppi_numbers.add(int(nums[1]))
            ppi_lines.append((int(nums[0]), int(nums[1])))  # 원본 저장

# Mapping 파일에서 왼쪽 값(Entrez ID)만 추출
mapping_numbers = set()
with open(mapping_file, "r") as f:
    for line in f:
        parts = line.strip().split(":")
        if len(parts) == 2:
            try:
                mapping_numbers.add(int(parts[1]))
            except ValueError:
                pass  # 숫자가 아닐 경우 무시

# PPI에서 매핑에 없는 숫자 찾기
missing_numbers = ppi_numbers - mapping_numbers

# 필터링하여 새로운 파일로 저장
with open(output_file, "w") as f:
    for num1, num2 in ppi_lines:
        if num1 not in missing_numbers and num2 not in missing_numbers:
            f.write(f"{num1}\t{num2}\n")

print(f"필터링된 PPI 파일이 '{output_file}'로 저장됨.")
