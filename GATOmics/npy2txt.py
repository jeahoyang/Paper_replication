import numpy as np

def npy_to_txt(npy_file, txt_file, delimiter="\t"):
    """
    npy 파일을 txt 파일로 변환하는 함수

    Parameters:
    - npy_file (str): 변환할 npy 파일 경로
    - txt_file (str): 저장할 txt 파일 경로
    - delimiter (str): 데이터 구분자 (기본값: 탭 '\t')
    """
    # npy 파일 로드
    data = np.load(npy_file)

    # txt 파일로 저장
    np.savetxt(txt_file, data, delimiter=delimiter, fmt="%.6f")

    print(f"변환 완료: {npy_file} -> {txt_file}")

# 사용 예시
npy_to_txt("Yn_sub_array_0.npy", "Yn_sub_array_0.txt")
