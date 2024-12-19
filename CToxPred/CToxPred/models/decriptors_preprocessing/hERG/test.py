import joblib
import numpy as np
from CToxPred.pairwise_correlation import CorrelationThreshold
import pandas as pd

# 저장된 파일 경로
file_path = 'herg_descriptors_preprocessing_pipeline.sav'

# 파일 로드
loaded_object = joblib.load(file_path)

# 객체의 타입 확인
print("Loaded Object Type:", type(loaded_object))

# 객체 내용 미리보기 (주로 데이터프레임이나 배열일 가능성)
if isinstance(loaded_object, (pd.DataFrame, np.ndarray)):
    print("Preview of Object:\n", loaded_object.head() if isinstance(loaded_object, pd.DataFrame) else loaded_object[:5])
elif isinstance(loaded_object, dict):
    print("Keys in Object:", loaded_object.keys())
else:
    print("Loaded Object Content:\n", loaded_object)

