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
from sklearn import linear_model
import torch.backends.cudnn as cudnn
import os

# Fixed seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True

EPOCH = 1  # Specific cancer training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(path, cancer_type):
    """ Load data for the specific cancer type or pan-cancer """
    network1 = []
    adj1 = sp.load_npz(path + "PP.adj.npz")
    adj2 = sp.load_npz(path + "PO.adj.npz")
    adj3 = sp.load_npz("./data/go.npz")
    adj4 = sp.load_npz("./data/exp.npz")

    network1.extend([adj1.tocoo(), adj2.tocoo(), adj3.tocoo(), adj4.tocoo()])
    
    network2 = []
    adj5 = sp.load_npz(path + "O.adj_loop.npz")
    adj6 = sp.load_npz(path + "O.N_all.npz")
    network2.extend([adj5.tocoo(), adj6.tocoo()])

    # 암 유형별 feature index
    cancer_dict = {
        'kirc': [0, 16, 32, 48], 'brca': [1, 17, 33, 49], 'prad': [3, 19, 35, 51],
        'stad': [4, 20, 36, 52], 'hnsc': [5, 21, 37, 53], 'luad': [6, 22, 38, 54],
        'thca': [7, 23, 39, 55], 'blca': [8, 24, 40, 56], 'esca': [9, 25, 41, 57],
        'lihc': [10, 26, 42, 58], 'ucec': [11, 27, 43, 59], 'coad': [12, 28, 44, 60],
        'lusc': [13, 29, 45, 61], 'cesc': [14, 30, 46, 62], 'kirp': [15, 31, 47, 63]
    }
    # cancer_dict = {
    #     'kirc': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
    # }
    

    # Load node features
    full_feat = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]
    full_feat = torch.Tensor(full_feat).to(device)

    # if cancer_type == 'pan-cancer':
    #     selected_indices = list(range(64))  # Pan-cancer uses all features
    # else:
    #     selected_indices = cancer_dict[cancer_type]  # Select features based on cancer type

    selected_indices = cancer_dict[cancer_type]
    
    # l_feature = [
    #     full_feat[:, selected_indices] for _ in range(3)
    # ] + [torch.Tensor(pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]).to(device)]  

    # r_feature = [
    #     full_feat[:, selected_indices],
    #     torch.Tensor(pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]).to(device),  # Outlying gene
    #     full_feat[:, selected_indices]  
    # ]
    
    # Load node features with slicing
    l_feature = []  # Gene features

    # Load full feature matrix
    full_feat = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]
    full_feat = torch.Tensor(full_feat).to(device)

    # 선택된 feature index 가져오기
    selected_indices = cancer_dict[cancer_type]

    # Gene feature 1
    feat1 = full_feat[:, selected_indices]  # 슬라이싱 적용
    l_feature.append(feat1)
    print('1')
    # Gene feature 2
    feat2 = full_feat[:, selected_indices]  # 슬라이싱 적용
    l_feature.append(feat2)
    print('2')
    # Gene feature 3
    feat3 = full_feat[:, selected_indices]  # 슬라이싱 적용
    l_feature.append(feat3)
    print('3')
    # Gene feature 4 (Full matrix 사용)
    feat4_exp = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]
    feat4_exp = torch.Tensor(feat4_exp).to(device)
    feat4_exp = feat4_exp[:, selected_indices]
    l_feature.append(feat4_exp)
    print('4')
    # -------------------------------
    # Load right-side features (r_feature)
    r_feature = []

    # Right-side Gene feature 1
    feat4 = full_feat[:, selected_indices]  # 슬라이싱 적용
    r_feature.append(feat4)
    print('5')
    # Outlying gene feature (Full matrix 사용)
    feat5 = pd.read_csv(path + "O.feat-final.csv", sep=",").values[:, 1:]
    feat5 = torch.Tensor(feat5).to(device)
    feat5 = feat5[:, selected_indices]
    r_feature.append(feat5)
    # print(feat5)
    print('6')
    # miRNA feature
    # feat6 = full_feat[:, selected_indices]  # 슬라이싱 적용
    r_feature.append(feat4)
    print('7')

    # Load edges
    pos_edge = np.array(np.loadtxt(path + "PP_pos.txt").transpose())
    pos_edge = torch.from_numpy(pos_edge).long()
    print('8')
    pb, _ = remove_self_loops(pos_edge)
    pos_edge1, _ = add_self_loops(pb)

    # Load k-fold validation sets
    with open(path + "/k_sets.pkl", 'rb') as handle:
        k_sets = pickle.load(handle)
    print('9')
    # Load labels
    label = np.loadtxt(path + "label_file.txt")
    Y = torch.tensor(label).float().to(device).unsqueeze(1)
    print('10')
    return network1, network2, l_feature, r_feature, pos_edge, pos_edge1, k_sets, Y


def train(mask, Y):
    model.train()
    optimizer.zero_grad()
    pred, pred1, r_loss, _ = model()
    loss = F.binary_cross_entropy_with_logits(pred[mask], Y[mask])
    loss += 0.1 * F.binary_cross_entropy_with_logits(pred1[mask], Y[mask]) + 0.01 * r_loss
    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(mask1, mask2, Y):
    model.eval()
    _, _, _, x = model()
    
    train_x, train_y = torch.sigmoid(x[mask1]).cpu().numpy(), Y[mask1].cpu().numpy()
    test_x, Yn = torch.sigmoid(x[mask2]).cpu().numpy(), Y[mask2].cpu().numpy().reshape(-1)
    
    # 로지스틱 회귀 모델을 학습
    pred = linear_model.LogisticRegression(max_iter=10000).fit(train_x, train_y.ravel()).predict_proba(test_x)[:, 1]

    # AUC
    auc_score = metrics.roc_auc_score(Yn, pred)

    # Precision-Recall Curve 계산
    precision, recall, _ = metrics.precision_recall_curve(Yn, pred)

    # recall 값이 단조 증가하도록 정렬
    sorted_indices = np.argsort(recall)
    recall_sorted = recall[sorted_indices]
    precision_sorted = precision[sorted_indices]

    # AUPRC 계산
    auprc_score = metrics.auc(recall_sorted, precision_sorted)

    # F1-score 계산
    f1_score = metrics.f1_score(Yn, (pred >= 0.5).astype(int))

    return auc_score, auprc_score, f1_score

if __name__ == '__main__':
    time_start = time.time()
    path = "./data/pan-cancer/"
    cancer_list = ['kirc', 'brca', 'prad', 'stad', 'hnsc', 'luad', 'thca', 'blca', 'esca', 'lihc', 'ucec', 'coad', 'lusc', 'cesc', 'kirp']
    for cancer in cancer_list:
        print(f"Training for {cancer}...")
        network1, network2, l_feature, r_feature, pos_edge, pos_edge1, _, Y = load_data(path, cancer)
        # print(l_feature)
        # print(r_feature)
        model = Net(l_feature, r_feature, network1, network2, 1, 4, 256, 128, pos_edge, pos_edge1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

        train_mask, test_mask = torch.ones(len(Y), dtype=bool), torch.zeros(len(Y), dtype=bool)
        train_mask[:int(len(Y) * 0.8)], test_mask[int(len(Y) * 0.8):] = True, True
        
        for epoch in range(1, EPOCH + 1):
            train(train_mask, Y)
            if epoch % 50 == 0:
                print(f"Epoch {epoch}")
                scheduler.step()

        torch.save(model, f'model_{cancer}.pth')
        AUC, AUPRC, F1 = test(train_mask, test_mask, Y)
        print(f"{cancer} - AUC: {AUC:.4f}, AUPRC: {AUPRC:.4f}, F1-score: {F1:.4f}")
    print(f"Total Time: {time.time() - time_start:.2f} sec")
