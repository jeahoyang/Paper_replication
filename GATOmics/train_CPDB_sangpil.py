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


# fixed seed
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True

# EPOCH = 50      # pan-cancer
EPOCH = 1000        # specific cancer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
def load_label_single(path):
    label = np.loadtxt(path + "label_file-P.txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)
    label_pos = np.loadtxt(path + "pos.txt", dtype=int)
    label_neg = np.loadtxt(path + "neg.txt", dtype=int)


    return Y, label_pos, label_neg

def sample_division_single(pos_label, neg_label, l, l1, l2, i):
    # pos_label：Positive sample index
    # neg_label：Negative sample index
    # l：number of genes
    # l1：Number of positive samples
    # l2：number of negative samples
    # i：number of folds
    pos_test = pos_label[i * l1:(i + 1) * l1]
    pos_train = list(set(pos_label) - set(pos_test))
    neg_test = neg_label[i * l2:(i + 1) * l2]
    neg_train = list(set(neg_label) - set(neg_test))
    indexs1 = [False] * l
    indexs2 = [False] * l
    for j in range(len(pos_train)):
        indexs1[pos_train[j]] = True
    for j in range(len(neg_train)):
        indexs1[neg_train[j]] = True
    for j in range(len(pos_test)):
        indexs2[pos_test[j]] = True
    for j in range(len(neg_test)):
        indexs2[neg_test[j]] = True
    tr_mask = torch.from_numpy(np.array(indexs1))
    te_mask = torch.from_numpy(np.array(indexs2))

    return tr_mask, te_mask

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

    # 其余代码保持不变...


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

def train(mask, Y):
    model.train()
    optimizer.zero_grad()

    pred, pred1, r_loss, _ = model()
    loss = F.binary_cross_entropy_with_logits(pred[mask], Y[mask])
    loss1 = F.binary_cross_entropy_with_logits(pred1[mask], Y[mask])
    loss = loss + 0.1 * loss1 + 0.01 * r_loss

    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(mask1, mask2, Y):
    model.eval()
    _, _, _, x = model()

    # logistic regression model
    train_x = torch.sigmoid(x[mask1]).cpu().detach().numpy()
    train_y = Y[mask1].cpu().numpy()
    test_x = torch.sigmoid(x[mask2]).cpu().detach().numpy()
    Yn = Y[mask2].cpu().numpy().reshape(-1)
    pred = LR(train_x, train_y, test_x)
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    
    # F1-score 계산 (0.5 threshold)
    pred_binary = (pred >= 0.5).astype(int)  # 0.5를 기준으로 이진 분류
    f1 = metrics.f1_score(Yn, pred_binary)

    return metrics.roc_auc_score(Yn, pred), area, f1, Yn, pred


fold_dir = "./10fold"
    
def load_fold_data(fold_idx):
    fold_path = os.path.join(fold_dir, f'fold_{fold_idx}')
    
    # .txt 파일을 읽어서 텐서로 변환하는 함수
    def load_data_from_txt(file_path):
        with open(file_path, 'r') as file:
            # 파일 내용을 읽어서 리스트로 변환 (각 줄을 int로 변환)
            data = [int(line.strip()) for line in file.readlines()]
        return torch.tensor(data, dtype=torch.long)

    # 데이터 로딩
    train_data = load_data_from_txt(os.path.join(fold_path, 'train.txt'))
    valid_data = load_data_from_txt(os.path.join(fold_path, 'valid.txt'))
    test_data = load_data_from_txt(os.path.join(fold_path, 'test.txt'))
    
    train_mask = load_data_from_txt(os.path.join(fold_path, 'train_mask.txt'))
    valid_mask = load_data_from_txt(os.path.join(fold_path, 'valid_mask.txt'))
    test_mask = load_data_from_txt(os.path.join(fold_path, 'test_mask.txt'))
    
    labels = load_data_from_txt(os.path.join(fold_path, 'labels.txt'))
    
    return train_data, valid_data, test_data, train_mask, valid_mask, test_mask, labels


def load_kfold_data(fold_path, device):
    """STRING_10fold 데이터셋의 특정 fold에서 데이터를 로드하는 함수"""
    # Load indices
    train_idx = np.loadtxt(f"{fold_path}/train.txt", dtype=int)
    valid_idx = np.loadtxt(f"{fold_path}/valid.txt", dtype=int)
    test_idx = np.loadtxt(f"{fold_path}/test.txt", dtype=int)
    
    # Load masks (0/1 또는 True/False)
    train_mask = torch.tensor(np.loadtxt(f"{fold_path}/train_mask.txt", dtype=bool), device=device)
    valid_mask = torch.tensor(np.loadtxt(f"{fold_path}/valid_mask.txt", dtype=bool), device=device)
    test_mask = torch.tensor(np.loadtxt(f"{fold_path}/test_mask.txt", dtype=bool), device=device)
    
    # Load labels
    labels = torch.tensor(np.loadtxt(f"{fold_path}/labels.txt"), dtype=torch.float32, device=device)
    
    return train_idx, valid_idx, test_idx, train_mask, valid_mask, test_mask, labels

def filter_network_by_genes(network_list, original_gene_file, target_gene_file):
    """
    Filters and reorders each adjacency matrix in network_list
    to keep only the genes listed in target_gene_file,
    using indices based on original_gene_file.

    Returns:
        - filtered_networks: list of adjacency matrices (shape MxM)
        - valid_genes_filtered: list of retained gene names (order preserved)
    """
    # Load gene name mapping
    with open(original_gene_file, 'r') as f:
        gene_map = [line.strip().split(',')[1] for line in f.readlines()]
    gene_to_idx = {g: i for i, g in enumerate(gene_map)}

    with open(target_gene_file, 'r') as f:
        target_genes = [line.strip() for line in f.readlines()]

    valid_genes = [g for g in target_genes if g in gene_to_idx]
    print(f"Initially matched {len(valid_genes)} valid genes out of {len(target_genes)}")

    # Map to indices once
    all_indices = [gene_to_idx[g] for g in valid_genes]

    filtered_networks = []
    final_valid_genes = []

    for i, adj in enumerate(network_list):
        adj = adj.tocsr()
        n = adj.shape[0]

        # 네트워크 범위 안의 인덱스만 필터링
        indices = [idx for idx in all_indices if idx < n]
        genes = [g for g in valid_genes if gene_to_idx[g] < n]

        if not indices:
            raise ValueError(f"No valid indices within bounds for network[{i}]")

        print(f"[Network {i}] Filtering to {len(indices)} genes (adj shape: {adj.shape})")
        sub_adj = adj[indices, :][:, indices]
        filtered_networks.append(sub_adj)

        if not final_valid_genes:
            final_valid_genes = genes
        else:
            final_valid_genes = [g for g in final_valid_genes if g in genes]

    return filtered_networks, final_valid_genes

def filter_row_network_by_genes(network_list, original_gene_file, target_gene_file):
    """
    Filters and reorders the *rows only* of each adjacency matrix in network_list,
    according to the genes listed in target_gene_file (based on original_gene_file indexing).

    Parameters:
        network_list: list of scipy sparse adjacency matrices (shape: gene × other)
        original_gene_file: path to gene_names.txt (ENSG...,GeneName)
        target_gene_file: path to feature_genename.txt (GeneName per line)

    Returns:
        - filtered_networks: list of filtered adjacency matrices with filtered rows
        - valid_genes: list of kept gene names in order of target_gene_file
    """
    # 1. Load original gene name mapping
    with open(original_gene_file, 'r') as f:
        gene_map = [line.strip().split(',')[1] for line in f.readlines()]
    gene_to_idx = {g: i for i, g in enumerate(gene_map)}

    # 2. Load target gene list
    with open(target_gene_file, 'r') as f:
        target_genes = [line.strip() for line in f.readlines()]

    # 3. Filter valid genes
    valid_genes = [g for g in target_genes if g in gene_to_idx]
    print(f"Filtered {len(valid_genes)} valid genes out of {len(target_genes)}")

    # 4. Get corresponding row indices
    indices = [gene_to_idx[g] for g in valid_genes]

    # 5. Filter each network (rows only)
    filtered_networks = []
    for i, adj in enumerate(network_list):
        print(f"[Network {i}] Original shape: {adj.shape}")
        adj = adj.tocsr()
        if max(indices) >= adj.shape[0]:
            raise ValueError(f"Index out of bounds in network[{i}]: max index {max(indices)}, adj shape {adj.shape}")
        sub_adj = adj[indices, :]  # rows only
        print(f"[Network {i}] Filtered shape: {sub_adj.shape}")
        filtered_networks.append(sub_adj)

    return filtered_networks, valid_genes

def filter_and_reorder_features_edges(l_feature, r_feature, pos_edge, pos_edge1,
                                      original_gene_file, target_gene_file):
    """
    Filters and reorders l_feature, r_feature[0], pos_edge, and pos_edge1
    based on gene names in feature_genename.txt

    Returns:
        - new_l_feature: list of filtered l_feature
        - new_r_feature: updated r_feature (only r_feature[0] is modified; others untouched)
        - new_pos_edge: filtered and reindexed pos_edge
        - new_pos_edge1: filtered and reindexed pos_edge1
        - valid_genes: ordered list of retained gene names
    """
    # 원본 유전자 이름 로드
    with open(original_gene_file, 'r') as f:
        gene_map = [line.strip().split(',')[1] for line in f.readlines()]
    gene_to_idx = {g: i for i, g in enumerate(gene_map)}

    # 사용할 유전자 이름 로드
    with open(target_gene_file, 'r') as f:
        target_genes = [line.strip() for line in f.readlines()]
    valid_genes = [g for g in target_genes if g in gene_to_idx]

    # 인덱스 재정렬용 매핑
    old_indices = [gene_to_idx[g] for g in valid_genes]
    old_idx_to_new = {old: new for new, old in enumerate(old_indices)}

    # ✅ l_feature 재정렬
    new_l_feature = [feat[old_indices] for feat in l_feature]

    # ✅ r_feature[0]만 재정렬, 나머지는 그대로 유지
    new_r_feature = [
        # r_feature[0][old_indices],  # gene → 필터링
        r_feature[0], # 일단 그대로 유지
        r_feature[1],               # outlying gene → 그대로 유지
        r_feature[2]                # miRNA → 그대로 유지
    ]
    
    # ✅ edge 필터링 및 재정렬
    def filter_and_remap_edges(edges):
        edge_list = []
        for u, v in edges.T:  # edges shape: [2, E]
            if int(u) in old_idx_to_new and int(v) in old_idx_to_new:
                u_new = old_idx_to_new[int(u)]
                v_new = old_idx_to_new[int(v)]
                edge_list.append((u_new, v_new))
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).T

    new_pos_edge = filter_and_remap_edges(pos_edge)
    new_pos_edge1 = filter_and_remap_edges(pos_edge1)

    return new_l_feature, new_r_feature, new_pos_edge, new_pos_edge1, valid_genes

# if __name__ == '__main__':
#     time_start = time.time()

#     # results
#     AUC_test = np.zeros(shape=(10,5))
#     AUPRC_test = np.zeros(shape=(10,5))

#     # load data
#     path = "./data/pan-cancer/"     # pan-cancer
#     # path = "./data/LUAD/cancer name/"     # specific cancer,e.g.luad
#     network1, network2, l_feature, r_feature, pos_edge, pos_edge1, k_sets, Y = load_data(path)

#     # 十次五倍交叉
#     # pan-cancer
#     for i in range(5):
#         print("\n", "times:", i, "\n")
#         Yn_sub_list = []
#         pred_sub_list = []
#         for cv_run in range(5):
#             print("the %s five-fold cross:\n" % cv_run)
#             _, _, tr_mask, te_mask = k_sets[i][cv_run]
#             # load model
#             model = Net(l_feature, r_feature, network1, network2, 1, 64, 256, 128, pos_edge, pos_edge1).to(device)    # hop=1
#             optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0005)
#             decayRate = 0.96
#             my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
#             for epoch in range(1,EPOCH+1):
#                 train(tr_mask, Y)
#                 if epoch % 50 ==0:
#                     print(epoch)
#                     my_lr_scheduler.step()
#             torch.save(model, 'model_'+str(i)+'_' +str(cv_run) + '.pth')
#             AUC, AUPRC, Yn_sub, pred_sub = test(tr_mask, te_mask, Y)
#             Yn_sub_list.append(Yn_sub)
#             pred_sub_list.append(pred_sub)
#             print(AUC)
#             print(AUPRC)
#             AUC_test[i][cv_run] = AUC
#             AUPRC_test[i][cv_run] = AUPRC
#             print(time.time() - time_start)
#             # 保存结果
#             np.savetxt("./AUC_test.txt", AUC_test, delimiter="\t")
#             np.savetxt("./AUPR_test.txt", AUPRC_test, delimiter="\t")
# #        Yn_sub_array = np.vstack(Yn_sub_list)
# #        pred_sub_array = np.vstack(pred_sub_list)
#             np.save('Yn_sub_array'+str(i)+'_'+str(cv_run)+'.npy', Yn_sub_list)
#             np.save('pred_sub_array'+str(i)+'_'+str(cv_run)+'.npy', pred_sub_list)



if __name__ == '__main__':
    time_start = time.time()

    # 결과 배열 초기화
    AUC_test = np.zeros(shape=(10, 1))
    AUPRC_test = np.zeros(shape=(10, 1))
    F1_test = np.zeros(shape=(10, 1))

    # 데이터 로드
    path = "./data/pan-cancer/"
    network1, network2, l_feature, r_feature, pos_edge, pos_edge1, _, Y = load_data(path)

    # 유전자 이름 불러오기
    gene_file = "./data/gene_names.txt"
    with open(gene_file, "r") as f:
        gene_names = [line.strip().split(',')[1] for line in f.readlines()]  # 첫 번째 열만 가져옴
    
    
    network1_except_1 = [network1[i] for i in range(len(network1)) if i != 1]
    network1_only_1 = [network1[1]]
    
    filtered_network1, filtered_gene_names = filter_network_by_genes(
        network1_except_1,  # gene-gene 관련 네트워크만 전달
        "./data/gene_names.txt",
        "./CPDB/feature_genename.txt"
    )
    
    filtered_network1_1, filtered_gene_names = filter_row_network_by_genes(
        network1_only_1,  # gene-outlying gene network (shape: N × M)
        "./data/gene_names.txt",
        "./CPDB/feature_genename.txt"
    )
    
    filtered_network_all = (
        [filtered_network1[0]] +
        [filtered_network1_1[0]] +  # 리스트에서 원소 꺼냄
        filtered_network1[1:]
    )
        
    # filtered_network2, _ = filter_network_by_genes(
    #     network2[:2],  # gene-gene 관련 네트워크만 전달
    #     "./data/gene_names.txt",
    #     "./CPDB/feature_genename.txt"
    # )
    
    print('check')
    print(len(filtered_gene_names))
    
    original_gene_file = "./data/gene_names.txt"
    target_gene_file = "./CPDB/feature_genename.txt"
    
    print("l_feature[0] shape BEFORE filtering:", l_feature[0].shape)
    print("r_feature[1] shape BEFORE filtering:", r_feature[1].shape)
    l_feature, new_r_feature, pos_edge, pos_edge1, filtered_genes = filter_and_reorder_features_edges(l_feature, r_feature, pos_edge, pos_edge1, original_gene_file, target_gene_file)
    print("r_feature[1] shape AFTER filtering:", new_r_feature[1].shape)
    print("l_feature[0] shape AFTER filtering:", l_feature[0].shape)
    cross_val = 10
    
    AUC = np.zeros(shape=(cross_val))
    AUPR = np.zeros(shape=(cross_val))
    F1_SCORES = np.zeros(cross_val)
    
    print("---------Pan-cancer Train begin--------")
    for i in range(cross_val):
        fold_path = f"./CPDB/CPDB_10fold/fold_{i+1}"  # 폴더 경로 설정
        train_idx, valid_idx, test_idx, train_mask, valid_mask, test_mask, Y = load_kfold_data(fold_path, device)
        
        # 모델 초기화
        model = Net(l_feature, new_r_feature, filtered_network_all, network2, 1, 64, 256, 128, pos_edge, pos_edge1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0005)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)

        # 모델 훈련
        for epoch in range(1, EPOCH + 1):
            train(train_mask, Y)  # train_mask 적용
            if epoch % 50 == 0:
                print(epoch)
                my_lr_scheduler.step()

        # 모델 저장
        torch.save(model, f'model_{i}.pth')

        # 테스트
        AUC[i], AUPR[i], F1_SCORES[i], Yn_sub, pred_sub = test(train_mask, test_mask, Y)
        
        print(f'(T) | TEST AUC={AUC[i]:.3f}, AUPR={AUPR[i]:.3f}, F1-score={F1_SCORES[i]:.3f}')

        # 각 fold 결과를 텍스트 파일에 기록
        with open("./Result_cpdb.txt", "a") as result_file:
            result_file.write(f"Fold {i+1}: AUC={AUC[i]:.3f}, AUPR={AUPR[i]:.3f}, F1-score={F1_SCORES[i]:.3f}\n")
            
    # 평균 및 표준편차 계산
    mean_auc = AUC.mean()
    std_auc = AUC.std()
    mean_aupr = AUPR.mean()
    std_aupr = AUPR.std()
    mean_f1 = F1_SCORES.mean()
    std_f1 = F1_SCORES.std()

    # 결과 출력
    print(f"\n🎯 최종 평균 AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"🎯 최종 평균 AUPR: {mean_aupr:.3f} ± {std_aupr:.3f}")
    print(f"🎯 최종 평균 F1-score: {mean_f1:.3f} ± {std_f1:.3f}")
    
    # 최종 결과 저장
    with open("./Result_final_cpdb.txt", "w") as f:
        f.write("Metric\tMean\tStd\n")
        f.write(f"AUROC\t{mean_auc:.4f}\t{std_auc:.4f}\n")
        f.write(f"AUPRC\t{mean_aupr:.4f}\t{std_aupr:.4f}\n")
        f.write(f"F1-score\t{mean_f1:.4f}\t{std_f1:.4f}\n")

    print("10-fold 교차 검증 최종 결과 저장 완료: final_results.txt")
    
    # # 10-fold 교차 검증
    # for i in range(10):
    #     print("\n", "fold:", i + 1, "\n")
    #     Yn_sub_list = []
    #     pred_sub_list = []

    #     # 폴더 경로 설정
    #     fold_path = f'10fold_CPDB/fold_{i+1}'
    #     test_mask_file = os.path.join(fold_path, 'test_mask.txt')
    #     train_mask_file = os.path.join(fold_path, 'train_mask.txt')

    #     # 마스크 파일 로드
    #     train_mask = np.loadtxt(train_mask_file).astype(bool)
    #     test_mask = np.loadtxt(test_mask_file).astype(bool)

    #     # 인덱스 범위 검사
    #     assert len(train_mask) == len(Y), "train_mask 크기가 Y와 다릅니다."
    #     assert len(test_mask) == len(Y), "test_mask 크기가 Y와 다릅니다."

    #     print(np.unique(Y[test_mask].cpu().numpy()))

    #     # 모델 초기화
    #     model = Net(l_feature, r_feature, network1, network2, 1, 64, 256, 128, pos_edge, pos_edge1).to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0005)
    #     my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)

    #     # 모델 훈련
    #     for epoch in range(1, EPOCH + 1):
    #         train(train_mask, Y)  # train_mask 적용
    #         if epoch % 50 == 0:
    #             print(epoch)
    #             my_lr_scheduler.step()

    #     # 모델 저장
    #     torch.save(model, f'model_{i}.pth')

    #     # 테스트
    #     AUC, AUPRC, f1, Yn_sub, pred_sub = test(train_mask, test_mask, Y)

    #     # 결과 기록
    #     print(f"AUC: {AUC:.4f}, AUPRC: {AUPRC:.4f}, F1-score: {f1:.4f}")
    #     AUC_test[i] = AUC
    #     AUPRC_test[i] = AUPRC
    #     F1_test[i] = f1

    #     print(f"Elapsed time: {time.time() - time_start:.2f} sec")

    #     # Fold별 결과 저장
    #     np.savetxt("./AUC_test.txt", AUC_test, delimiter="\t")
    #     np.savetxt("./AUPRC_test.txt", AUPRC_test, delimiter="\t")
    #     np.savetxt("./F1_test.txt", F1_test, delimiter="\t")

    #     # 예측 결과 저장 (txt 파일)
    #     output_file = f'predictions_fold_{i+1}.txt'
    #     with open(output_file, 'w') as f:
    #         f.write("Gene\tPredicted\tLabel\tF1-score\n")
    #         for gene, p, l in zip(np.array(gene_names)[test_mask], pred_sub, Yn_sub):
    #             f.write(f"{gene}\t{p:.6f}\t{l}\t{f1:.4f}\n")

    #     print(f"예측 결과 저장 완료: {output_file}")

    # # 최종 결과 계산 (10-fold 평균 및 표준편차)
    # AUC_mean, AUC_std = np.mean(AUC_test), np.std(AUC_test)
    # AUPRC_mean, AUPRC_std = np.mean(AUPRC_test), np.std(AUPRC_test)
    # F1_mean, F1_std = np.mean(F1_test), np.std(F1_test)

    # # 최종 결과 저장
    # with open("final_results.txt", "w") as f:
    #     f.write("Metric\tMean\tStd\n")
    #     f.write(f"AUROC\t{AUC_mean:.4f}\t{AUC_std:.4f}\n")
    #     f.write(f"AUPRC\t{AUPRC_mean:.4f}\t{AUPRC_std:.4f}\n")
    #     f.write(f"F1-score\t{F1_mean:.4f}\t{F1_std:.4f}\n")

    # print("10-fold 교차 검증 최종 결과 저장 완료: final_results.txt")
