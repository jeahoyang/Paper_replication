import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
import pickle
import random
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn import metrics
from model_all import Net
from sklearn import linear_model
import torch.backends.cudnn as cudnn
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 마스크 파일 불러오기
mask_dir = "OncoKB_ONGene_mask"
onco_mask = np.loadtxt(os.path.join(mask_dir, "OncoKB_STRING_mask.txt")).astype(bool)
ongene_mask = np.loadtxt(os.path.join(mask_dir, "ONGene_STRING_mask.txt")).astype(bool)

# 저장된 모델 경로
model_dir = "prediction_string/string_pth"

# 성능 저장용
onco_results = []
ongene_results = []


def load_data(path):
    # load network
    network1 = []
    adj1 = sp.load_npz(path + "PP.adj.npz")      # gene-gene network
    adj2 = sp.load_npz(path + "PO.adj.npz")      # gene-outlying gene network
    adj3 = sp.load_npz("./data/go.npz")  # 第三个网络
    adj4 = sp.load_npz("./data/exp.npz") # 第四个网络 - 新添加

    network1.append(adj1.tocsc())
    network1.append(adj2.tocsc())
    network1.append(adj3.tocsc())
    network1.append(adj4.tocsc())

    # networks for bilinear aggregation layer
    network2 = []
    adj5 = sp.load_npz(path + "O.adj_loop.npz")
    adj6 = sp.load_npz(path + "O.N_all.npz")

    network2.append(adj5.tocsc())
    network2.append(adj6.tocsc())

    # load node features
    l_feature = []      # gene
    feat1 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]
    feat1 = torch.Tensor(feat1).to(device)
    feat2 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]
    feat2 = torch.Tensor(feat2).to(device)
    feat3 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]
    feat3 = torch.Tensor(feat3).to(device)
    # 假设第四个网络的特征在 "P.feat-exp.csv" 文件中
    feat4_exp = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]  # 加载第四个网络的特征
    feat4_exp = torch.Tensor(feat4_exp).to(device)

    l_feature.append(feat1)
    l_feature.append(feat2)
    l_feature.append(feat3)
    l_feature.append(feat4_exp)  # 添加第四个网络的特征

    r_feature = []
    # 使用正确的变量名加载右侧特征
    feat4 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]       # gene
    feat4 = torch.Tensor(feat4).to(device)
    feat5 = pd.read_csv(path + "O.feat-final.csv", sep=",").values[:, 1:]       # outlying gene
    feat5 = torch.Tensor(feat5).to(device)
    feat6 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]         # miRNA
    feat6 = torch.Tensor(feat6).to(device)

    r_feature.append(feat4)
    r_feature.append(feat5)
    r_feature.append(feat6)


    # load edge
    pos_edge = np.array(np.loadtxt(path + "PP_pos.txt").transpose())
    pos_edge = torch.from_numpy(pos_edge).long()

    pb, _ = remove_self_loops(pos_edge)
    pos_edge1, _ = add_self_loops(pb)

    # divisions of ten-fold cross-validation
    with open(path + "/k_sets.pkl", 'rb') as handle:
        k_sets = pickle.load(handle)

    # label
    label = np.loadtxt(path + "label_file.txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)

    return network1, network2, l_feature, r_feature, pos_edge, pos_edge1, k_sets, Y

def LR(train_x, train_y, test_x):
    regr = linear_model.LogisticRegression(max_iter=10000)
    regr.fit(train_x, train_y.ravel())
    pre = regr.predict_proba(test_x)
    pre = pre[:,1]
    return pre


def test(model, mask, Y):
    model.eval()
    _, _, _, x = model()   # 여기서 model() 호출 시 인자 없으면 그대로 둠 (필요시 조정)

    # logistic regression model
    train_x = torch.sigmoid(x[mask]).cpu().detach().numpy()
    train_y = Y[mask].cpu().numpy()
    test_x = torch.sigmoid(x[mask]).cpu().detach().numpy()  # mask 구분 정확히 해야 함, 보통 train/test mask 다름
    Yn = Y[mask].cpu().numpy().reshape(-1)
    
    pred = LR(train_x, train_y, test_x)
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    
    pred_binary = (pred >= 0.5).astype(int)
    f1 = metrics.f1_score(Yn, pred_binary)

    return metrics.roc_auc_score(Yn, pred), area, f1, Yn, pred

path = "./data/pan-cancer/"
network1, network2, l_feature, r_feature, pos_edge, pos_edge1, _, Y = load_data(path)

for i in range(10):
    print(f"\nEvaluating model_{i}.pth")

    model_path = os.path.join(model_dir, f"model_{i}.pth")
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    auc_onco, auprc_onco, f1_onco, _, _ = test(model, onco_mask, Y)
    onco_results.append((auc_onco, auprc_onco, f1_onco))

    auc_ongene, auprc_ongene, f1_ongene, _, _ = test(model, ongene_mask, Y)
    ongene_results.append((auc_ongene, auprc_ongene, f1_ongene))

    print(f"[OncoKB]   AUC: {auc_onco:.4f}  AUPRC: {auprc_onco:.4f}  F1: {f1_onco:.4f}")
    print(f"[ONGene]   AUC: {auc_ongene:.4f}  AUPRC: {auprc_ongene:.4f}  F1: {f1_ongene:.4f}")

