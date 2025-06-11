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
    
    feature_genename_file = './feature_genename.txt'
    
    # 10-fold 교차 검증
    for i in range(10):
        print("\n", "fold:", i + 1, "\n")
        Yn_sub_list = []
        pred_sub_list = []
        
        # 폴더 경로 설정
        fold_path = f'10fold_CPDB/fold_{i+1}'
        test_mask_file = os.path.join(fold_path, 'test_mask.txt')
        train_mask_file = os.path.join(fold_path, 'train_mask.txt')

        # 마스크 파일 로드
        train_mask = np.loadtxt(train_mask_file).astype(bool)
        test_mask = np.loadtxt(test_mask_file).astype(bool)

        # 인덱스 범위 검사
        assert len(train_mask) == len(Y), "train_mask 크기가 Y와 다릅니다."
        assert len(test_mask) == len(Y), "test_mask 크기가 Y와 다릅니다."

        print(np.unique(Y[test_mask].cpu().numpy()))

        # 모델 초기화
        model = Net(l_feature, r_feature, network1, network2, 1, 64, 256, 128, pos_edge, pos_edge1).to(device)
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
        AUC, AUPRC, f1, Yn_sub, pred_sub = test(train_mask, test_mask, Y)

        # 결과 기록
        print(f"AUC: {AUC:.4f}, AUPRC: {AUPRC:.4f}, F1-score: {f1:.4f}")
        AUC_test[i] = AUC
        AUPRC_test[i] = AUPRC
        F1_test[i] = f1

        print(f"Elapsed time: {time.time() - time_start:.2f} sec")

        # Fold별 결과 저장
        np.savetxt("./AUC_test.txt", AUC_test, delimiter="\t")
        np.savetxt("./AUPRC_test.txt", AUPRC_test, delimiter="\t")
        np.savetxt("./F1_test.txt", F1_test, delimiter="\t")

        # 예측 결과 저장 (txt 파일)
        output_file = f'predictions_fold_{i+1}.txt'
        with open(output_file, 'w') as f:
            f.write("Gene\tPredicted\tLabel\tF1-score\n")
            for gene, p, l in zip(np.array(gene_names)[test_mask], pred_sub, Yn_sub):
                f.write(f"{gene}\t{p:.6f}\t{l}\t{f1:.4f}\n")

        print(f"예측 결과 저장 완료: {output_file}")

    # 최종 결과 계산 (10-fold 평균 및 표준편차)
    AUC_mean, AUC_std = np.mean(AUC_test), np.std(AUC_test)
    AUPRC_mean, AUPRC_std = np.mean(AUPRC_test), np.std(AUPRC_test)
    F1_mean, F1_std = np.mean(F1_test), np.std(F1_test)

    # 최종 결과 저장
    with open("final_results.txt", "w") as f:
        f.write("Metric\tMean\tStd\n")
        f.write(f"AUROC\t{AUC_mean:.4f}\t{AUC_std:.4f}\n")
        f.write(f"AUPRC\t{AUPRC_mean:.4f}\t{AUPRC_std:.4f}\n")
        f.write(f"F1-score\t{F1_mean:.4f}\t{F1_std:.4f}\n")

    print("10-fold 교차 검증 최종 결과 저장 완료: final_results.txt")
