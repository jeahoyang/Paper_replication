import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F


# hERGClassifier 정의
class hERGClassifier(torch.nn.Module):
    """
    hERG Classifier architecture to process fingerprints.

    Parameters:
        inputSize : int
            The input size of the fingerprints representations.
        output_dim : int
            The output dimension of the classifier.
    """

    def __init__(self, input_size, output_size, dropout_rate=0.2):
        super(hERGClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 200, bias=True)
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        #self.bn1 = torch.nn.BatchNorm1d(num_features=200)

        self.linear2 = torch.nn.Linear(200, 200, bias=True)
        torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity="relu")
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        #self.bn2 = torch.nn.BatchNorm1d(num_features=200)

        self.linear3 = torch.nn.Linear(200, output_size, bias=True)
        torch.nn.init.kaiming_normal_(self.linear3.weight, nonlinearity="relu")

    def forward(self, x):
        out = self.dropout1(F.relu(self.linear1(x)))
        out = self.dropout2(F.relu(self.linear2(out)))
        out = F.softmax(self.linear3(out), dim=1)
        return out

    def save(self, path):
        """
        Save model with its parameters to the given path.
        Conventionally the path should end with "*.model".

        Inputs:
        - path: path string
        """
        print("Saving model... %s" % path)
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load model dictionary. The
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        device = torch.device("cpu")
        self.load_state_dict(torch.load(path, map_location=device))



# SMILES를 Morgan Fingerprint로 변환하는 함수
def smiles_to_fingerprint(smiles, radius=2, nBits=1905):
    """
    SMILES 문자열을 Morgan Fingerprint로 변환
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    else:
        return None




# 데이터 로드 및 전처리
def load_data(file_path):
    data = pd.read_csv(file_path, sep="\t", header=None, names=["SMILES", "Label"])
    fingerprints = []
    labels = []

    for _, row in data.iterrows():
        fp = smiles_to_fingerprint(row["SMILES"])
        if fp:  # 유효한 SMILES만 처리
            fingerprints.append(torch.tensor(list(fp), dtype=torch.float32))
            labels.append(row["Label"])

    x_data = torch.stack(fingerprints)
    y_data = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x_data, y_data)


# 파일 경로 정의
train_file_path = r"C:\Users\ADS_Lab\Desktop\JH\Paper_Replication\CToxPred2\dataset\SkinReaction_train_no_salt.txt"
test_file_path = r"C:\Users\ADS_Lab\Desktop\JH\Paper_Replication\CToxPred2\dataset\SkinReaction_test_no_salt.txt"

# 학습 데이터 로드
train_dataset = load_data(train_file_path)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 테스트 데이터 로드
test_dataset = load_data(test_file_path)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 모델 초기화
inputSize = train_dataset[0][0].shape[0]  # Morgan Fingerprint 크기 (2048)
outputSize = len(set([y.item() for _, y in train_dataset]))  # 클래스 개수
model = hERGClassifier(input_size=inputSize, output_size=outputSize)

# 손실 함수 및 옵티마이저 정의
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 저장 및 성능 추적
best_auc = 0.0  # AUC 초기화
best_epoch = 0  # 가장 성능 좋은 모델의 에폭
best_model_path = "best_SkinReaction_classifier_trained.model"  # 가장 성능 좋은 모델 파일 경로

# AUC 계산 함수
def calculate_auc(model, loader):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            outputs = model(x_batch)
            preds = outputs[:, 1]  # 양성 클래스 확률
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

    auc = roc_auc_score(all_labels, all_preds)
    return auc

from sklearn.metrics import precision_recall_curve, auc

def calculate_aupr(model, loader):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            outputs = model(x_batch)
            preds = outputs[:, 1]  # 양성 클래스 확률
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    aupr_value = auc(recall, precision)  # Precision-Recall Curve로 AUPR 계산
    return aupr_value


# 학습
# 학습
epochs = 100  # 에포크 100으로 설정
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        x_batch, y_batch = batch
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 테스트 데이터에 대해 AUC 계산
    auc_value = calculate_auc(model, test_loader)
    print(f"Epoch {epoch + 1}/{epochs}, AUC: {auc_value}")

    # 성능이 가장 좋은 모델 저장
    if auc_value > best_auc:
        best_auc = auc_value
        best_epoch = epoch + 1  # 가장 성능 좋은 모델의 에폭
        torch.save(model.state_dict(), best_model_path)  # AUC 값이 가장 높은 모델 저장

        # 가장 좋은 모델에서 AUPR 계산
        best_aupr = calculate_aupr(model, test_loader)  # AUPR 계산

    model.eval()  # 평가 모드로 전환
    with torch.no_grad():
        # 파일명에 에폭 번호 포함
        output_file = f"./SkinReaction/epoch_{epoch+1}_predictions.txt"
        with open(output_file, 'w') as f:
            f.write("label\tpred(score)\n")  # 헤더 작성
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch)
                preds = outputs[:, 1]  # 양성 클래스 확률
                for label, pred in zip(y_batch.numpy(), preds.numpy()):
                    f.write(f"{label}\t{pred}\n")  # label과 pred 값을 파일에 저장

# 가장 성능 좋은 모델의 에폭, AUC, AUPR 출력
print(f"Best model achieved at epoch {best_epoch} with AUC: {best_auc} and AUPR: {best_aupr}")
print(f"Best model saved at: {best_model_path}")
