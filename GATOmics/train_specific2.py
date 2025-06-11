import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn import metrics
from model_all import Net
import torch.backends.cudnn as cudnn
import os
from sklearn import linear_model

# Fixed seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True

EPOCH = 1000  # Specific cancer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cancer_dict = {
    'kirc': [0, 16, 32, 48], 'brca': [1, 17, 33, 49], 'prad': [3, 19, 35, 51],
    'stad': [4, 20, 36, 52], 'hnsc': [5, 21, 37, 53], 'luad': [6, 22, 38, 54],
    'thca': [7, 23, 39, 55], 'blca': [8, 24, 40, 56], 'esca': [9, 25, 41, 57],
    'lihc': [10, 26, 42, 58], 'ucec': [11, 27, 43, 59], 'coad': [12, 28, 44, 60],
    'lusc': [13, 29, 45, 61], 'cesc': [14, 30, 46, 62], 'kirp': [15, 31, 47, 63]
}


cancer_dict2 = {
    'kirc': [0, 16], 'brca': [1, 17], 'prad': [3, 19], 'stad': [4, 20],
    'hnsc': [5, 21], 'luad': [6, 22], 'thca': [7, 23], 'blca': [8, 24],
    'esca': [9, 25], 'lihc': [10, 26], 'ucec': [11, 27], 'coad': [12, 28],
    'lusc': [13, 29], 'cesc': [14, 30], 'kirp': [15, 31]
}


def load_data(path, cancer_type):
    network1 = [
        sp.load_npz(path + "PP.adj.npz").tocsc(),
        sp.load_npz(path + "PO.adj.npz").tocsc(),
        sp.load_npz("./data/go.npz").tocsc(),
        sp.load_npz("./data/exp.npz").tocsc()
    ]

    network2 = [
        sp.load_npz(path + "O.adj_loop.npz").tocsc(),
        sp.load_npz(path + "O.N_all.npz").tocsc()
    ]

    cancer_indices = cancer_dict[cancer_type]
    cancer_indices2 = cancer_dict2[cancer_type]
    features = torch.Tensor(pd.read_csv(path + "P.feat-final.csv", sep="," ).values[:, 1:]).to(device)
    
    # p_feat = pd.read_csv(path + "P.feat-final.csv")
    # print(f"[P.feat] 차원: {p_feat.shape}")  # (샘플 수, 65) 출력 확인 (인덱스 열 포함)
    # o_feat = pd.read_csv(path + "O.feat-final.csv")
    # print(f"[O.feat] 차원: {o_feat.shape}")  # ← 핵심 확인 포인트
    
    l_feature = [features[:, cancer_indices] for _ in range(4)]
    # for i, lf in enumerate(l_feature):
    #     print(f"l_feature[{i}].shape: {lf.shape}")
    r_feature = [
        features[:, cancer_indices],
        torch.Tensor(pd.read_csv(path + "O.feat-final.csv", sep="," ).values[:, 1:]).to(device)[:, cancer_indices2],
        features[:, cancer_indices]
    ]
    # for i, lf in enumerate(r_feature):
    #     print(f"l_feature[{i}].shape: {lf.shape}")

    pos_edge = torch.from_numpy(np.loadtxt(path + "PP_pos.txt").transpose()).long()
    pos_edge1, _ = add_self_loops(remove_self_loops(pos_edge)[0])

    with open(path + "/k_sets.pkl", 'rb') as handle:
        k_sets = pickle.load(handle)

    # label = np.loadtxt(path + "label_file.txt")
    # Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)

    return network1, network2, l_feature, r_feature, pos_edge, pos_edge1, k_sets

def load_label_single(path, cancer_type):
    label = np.loadtxt(path + f"label_file-P-{cancer_type}.txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)
    label_pos = np.loadtxt(path + f"pos-{cancer_type}.txt", dtype=int)
    label_neg = np.loadtxt(path + "pan-neg.txt", dtype=int)
    return Y, label_pos, label_neg

def load_gene_name_list(gene_name_path):
    gene_info = pd.read_csv(gene_name_path, header=None)
    gene_names = gene_info[1].tolist()  # 순서대로 리스트화 (index는 생략)
    return gene_names


def stratified_kfold_split_10fold(pos_label, neg_label, l, l1, l2):
    folds = []
    for i in range(10):
        pos_test = pos_label[i * l1:(i + 1) * l1]
        pos_train = list(set(pos_label) - set(pos_test))
        neg_test = neg_label[i * l2:(i + 1) * l2]
        neg_train = list(set(neg_label) - set(neg_test))
        
        if len(pos_train) == 0 or len(neg_train) == 0:
            raise ValueError(f"Empty training set in fold {i+1}. Adjust split strategy.")
        
        indexs1 = [False] * l
        indexs2 = [False] * l
        
        for j in pos_train:
            indexs1[j] = True
        for j in neg_train:
            indexs1[j] = True
        for j in pos_test:
            indexs2[j] = True
        for j in neg_test:
            indexs2[j] = True
        
        tr_mask = torch.from_numpy(np.array(indexs1))
        te_mask = torch.from_numpy(np.array(indexs2))
        
        folds.append((tr_mask, te_mask))
    
    return folds

def train(mask, Y, model, optimizer):
    model.train()
    optimizer.zero_grad()
    
    pred, pred1, r_loss, _ = model()
    loss = F.binary_cross_entropy_with_logits(pred[mask], Y[mask])
    loss1 = F.binary_cross_entropy_with_logits(pred1[mask], Y[mask])
    loss = loss + 0.1 * loss1 + 0.01 * r_loss
    
    loss.backward()
    optimizer.step()

def LR(train_x, train_y, test_x):
    regr = linear_model.LogisticRegression(max_iter=10000)
    regr.fit(train_x, train_y.ravel())
    pre = regr.predict_proba(test_x)
    return pre[:,1]

# @torch.no_grad()
# def test(mask1, mask2, Y, model):
#     model.eval()
#     _, _, _, x = model()
    
#     train_x = torch.sigmoid(x[mask1]).cpu().detach().numpy()
#     train_y = Y[mask1].cpu().numpy()
#     test_x = torch.sigmoid(x[mask2]).cpu().detach().numpy()
#     Yn = Y[mask2].cpu().numpy().reshape(-1)
#     pred = LR(train_x, train_y, test_x)
#     precision, recall, _ = metrics.precision_recall_curve(Yn, pred)
#     area = metrics.auc(recall, precision)
    
#     pred_binary = (pred >= 0.5).astype(int)
#     f1 = metrics.f1_score(Yn, pred_binary)
    
#     return metrics.roc_auc_score(Yn, pred), area, f1, Yn, pred

@torch.no_grad()
def test(mask1, mask2, Y, model, gene_name_list):
    model.eval()
    _, _, _, x = model()
    
    train_x = torch.sigmoid(x[mask1]).cpu().numpy()
    train_y = Y[mask1].cpu().numpy()
    test_x = torch.sigmoid(x[mask2]).cpu().numpy()
    Yn = Y[mask2].cpu().numpy().reshape(-1)
    pred = LR(train_x, train_y, test_x)

    precision, recall, _ = metrics.precision_recall_curve(Yn, pred)
    auc_pr = metrics.auc(recall, precision)
    auc_roc = metrics.roc_auc_score(Yn, pred)
    pred_binary = (pred >= 0.5).astype(int)
    f1 = metrics.f1_score(Yn, pred_binary)

    indices = mask2.nonzero(as_tuple=True)[0].cpu().numpy()
    gene_names_selected = [gene_name_list[i] for i in indices]

    gene_result = pd.DataFrame({
        "Gene": gene_names_selected,
        "Predicted": pred,
        "Label": Yn,
        "F1-score": [f1] * len(pred)
    })

    return auc_roc, auc_pr, f1, gene_result


import os

if __name__ == '__main__':
    time_start = time.time()
    path = "specific-cancer-updated/"
    path2 = "data/pan-cancer/"
    result_dir = "specific-cancer-result2/"

    os.makedirs(result_dir, exist_ok=True)

    results = []

    gene_name_list = load_gene_name_list("data/gene_names.txt")
    
    for cancer_type in cancer_dict.keys():
        print(f"Processing {cancer_type}...")

        Y, label_pos, label_neg = load_label_single(path, cancer_type)
        np.random.shuffle(label_pos)
        np.random.shuffle(label_neg)
        network1, network2, l_feature, r_feature, pos_edge, pos_edge1, k_sets = load_data(path2, cancer_type)

        folds = stratified_kfold_split_10fold(
            label_pos, label_neg, len(Y), len(label_pos) // 10, len(label_neg) // 10
        )

        # 결과 저장 파일
        result_file = os.path.join(result_dir, f"results_{cancer_type}.txt")
        with open(result_file, "w") as f:
            f.write(f"Results for {cancer_type}\n")
            f.write("Fold,AUC-ROC,AUC-PR,F1\n")

        for fold_index, (train_mask, test_mask) in enumerate(folds):
            print(f"Fold {fold_index+1}/10 for {cancer_type}")

            # 모델 초기화
            model = Net(
                l_feature, r_feature, network1, network2,
                hop=1, inputdims=4, hiddendims=256,
                outputdims=128, edge_index=pos_edge, edge_index1=pos_edge1
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0005)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

            for epoch in range(1, EPOCH + 1):
                train(train_mask, Y, model, optimizer)
                if epoch % 50 == 0:
                    print(f"Epoch {epoch} completed")
                    my_lr_scheduler.step()

            # 모델 저장
            model_path = os.path.join(result_dir, f'model_{cancer_type}_fold{fold_index}.pth')
            torch.save(model, model_path)
            print(f"Model saved: {model_path}")

            # 모델 테스트
            # auc_roc, auc_pr, f1_score, y_true, y_pred = test(train_mask, test_mask, Y, model)
            auc_roc, auc_pr, f1_score, gene_result = test(train_mask, test_mask, Y, model, gene_name_list)

            print(f"  AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}, F1: {f1_score:.4f}")

            # 결과 저장
            with open(result_file, "a") as f:
                f.write(f"{fold_index+1},{auc_roc:.4f},{auc_pr:.4f},{f1_score:.4f}\n")
                
            # gene-level 결과 저장 (추가된 부분)
            gene_result_file = os.path.join(result_dir, f"gene_result_{cancer_type}_fold{fold_index+1}.csv")
            gene_result.to_csv(gene_result_file, index=False)
                        # 결과 리스트에 저장 (이 줄도 fold 루프 안에 넣어줘야 해)
            results.append((auc_roc, auc_pr, f1_score))  


        # fold 루프 끝난 후, 평균/표준편차 정리해서 파일에 추가
        results_np = np.array(results)
        mean_vals = results_np.mean(axis=0)
        std_vals = results_np.std(axis=0)

        with open(result_file, "a") as f:
            f.write("\n")
            f.write(f"Mean,AUC-ROC: {mean_vals[0]:.4f}, AUC-PR: {mean_vals[1]:.4f}, F1: {mean_vals[2]:.4f}\n")
            f.write(f"Std ,AUC-ROC: {std_vals[0]:.4f}, AUC-PR: {std_vals[1]:.4f}, F1: {std_vals[2]:.4f}\n")

        # 🔄 results 리스트 초기화 (다음 cancer_type 위해)
        results = []
        
    print("All training and testing complete. Results saved for each cancer type.")
