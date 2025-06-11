import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz  # load_npz를 추가

# txt 파일 경로
txt_file = "my/filtered/go.txt"  # 입력 파일 경로
npz_file = "my/npz/go.npz"  # 출력 파일 경로

# 텍스트 파일에서 데이터를 읽어 numpy 배열로 변환
data = np.loadtxt(txt_file, dtype=int)

# 왼쪽과 오른쪽 열로 데이터를 나눔
rows = data[:, 0]  # 첫 번째 열: 행 인덱스
cols = data[:, 1]  # 두 번째 열: 열 인덱스

# 가중치 값은 없으므로, 기본값으로 1을 사용 (가중치가 필요 없다면)
values = np.ones(data.shape[0])

# 행렬의 크기 추정 (가장 큰 인덱스를 사용)
num_rows = np.max(rows) + 1
num_cols = np.max(cols) + 1

# CSR 형식의 희소 행렬 생성
sparse_matrix = csr_matrix((values, (rows, cols)), shape=(num_rows, num_cols))

# 희소 행렬을 npz 파일로 저장
save_npz(npz_file, sparse_matrix)

# 변환된 npz 파일 확인 (선택 사항)
loaded_sparse_matrix = load_npz(npz_file)  # load_npz로 불러오기
print(loaded_sparse_matrix)  # 희소 행렬 출력
