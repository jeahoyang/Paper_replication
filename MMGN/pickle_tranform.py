import pickle

# 파일명 설정
pkl_file = "mapping_dict_EIDtoIndex.pickle"  # 원본 pickle 파일
txt_file = "mapping_dict_EIDtoIndex.txt"  # 변환된 텍스트 파일

# pickle 파일 로드
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

# 텍스트 파일로 저장
with open(txt_file, "w", encoding="utf-8") as f:
    for key, value in data.items():  # 딕셔너리라면 key-value 형식으로 저장
        f.write(f"{key}: {value}\n")

print(f"✅ 변환 완료: {txt_file}")
