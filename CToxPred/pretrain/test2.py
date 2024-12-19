import joblib
import numpy as np
from CToxPred.pairwise_correlation import CorrelationThreshold
from CToxPred.utils import compute_descriptor_features
import pandas as pd

# 데이터 파일 경로
# file_path = 'C:\\Users\\ADS_Lab\\Desktop\\JH\\Paper_Replication\\CToxPred\\dataset\\SkinReaction_train_no_salt.txt'
file_path = 'C:\\Users\\ADS_Lab\\Desktop\\JH\\Paper_Replication\\CToxPred\\data\\raw\\Cav1.2\\data_cav_dev.csv'

data_cav_dev = file_path
smiles_list = data_cav_dev['SMILES']

# Descriptor 계산
descriptors = compute_descriptor_features(smiles_list)

# 파이프라인 파일 경로
pipeline_path = 'C:\\Users\\ADS_Lab\\Desktop\\JH\\Paper_Replication\\CToxPred\\CToxPred\\models\\decriptors_preprocessing\\Cav1.2\\cav_descriptors_preprocessing_pipeline.sav'


# 파이프라인 로드
preprocessing_pipeline = joblib.load(pipeline_path)

# 새로운 데이터셋 로드
new_data = pd.read_csv(file_path)

# 데이터 확인
print(new_data.head())

# 파이프라인을 사용해 새로운 데이터셋 변환
transformed_data = preprocessing_pipeline.transform(new_data)

# 변환된 데이터 출력
print("Transformed Data:")
print(transformed_data)
