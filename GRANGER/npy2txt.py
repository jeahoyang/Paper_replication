import numpy as np

# 1. 파일 로드
data = np.load('example_data/mCAD-2000-1/time_output.npy')

# 2. 데이터 구조 확인 (선택)
print("Data shape:", data.shape)
print("First few values:\n", data[:5])

# 3. 텍스트 파일로 저장
np.savetxt('example_data/mCAD-2000-1/time_output.txt', data, fmt='%.6f')
