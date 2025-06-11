import numpy as np

# .npz 파일 경로 지정
file_path = "data\exp.npz"

# .npz 파일 열기
data = np.load(file_path)

# 'row'와 'col' 배열을 각각 가져오기
row = data['row'].astype(int)
col = data['col'].astype(int)

# 'row'와 'col' 배열을 합침
combined_array = np.column_stack((row, col))

# 'row col' 헤더를 추가하여 파일로 저장
output_path = "exp.txt"
header = "row\tcol"  # 헤더는 탭으로 구분

# 텍스트 파일로 저장 (헤더 포함)
np.savetxt(output_path, combined_array, fmt='%d', delimiter="\t", header=header, comments='')

print(f"Saved combined data to {output_path}")