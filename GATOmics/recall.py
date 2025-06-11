import pandas as pd
from sklearn.metrics import recall_score

recall_list = []

for fold in range(1, 11):
    print(f"\n== Fold {fold} ==")
    
    # 파일 경로
    file_path = f'OncoKB_string/predictions_fold_{fold}.txt'
    
    # 파일 읽기 (탭 구분자)
    df = pd.read_csv(file_path, sep='\t')
    
    # 예측값과 실제값
    pred_prob = df['Predicted'].values
    labels = df['Label'].values.astype(int)
    
    # threshold 0.5 기준 binary prediction
    pred_binary = (pred_prob >= 0.5).astype(int)
    
    # Recall 계산
    recall = recall_score(labels, pred_binary)
    recall_list.append(recall)
    
    print(f"Recall@0.5 = {recall:.4f}")

# 전체 평균 Recall
mean_recall = sum(recall_list) / len(recall_list)
print(f"\n=========================")
print(f"Mean Recall over 10 folds = {mean_recall:.4f}")
print(f"=========================")
