import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.datasets import Planetoid
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import argparse
# import yaml
# from yaml import SafeLoader
from utils import processingIncidenceMatrix, get_laplacian_positional_encoding
from model14 import Graphomer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
# from tqdm import tqdm
import scipy.sparse as sp
import networkx as nx
# from torch_geometric.utils import to_networkx
import networkx as nx
# from scipy import sparse
# from scipy.sparse.csgraph import shortest_path
import gc
import os


# 1. 하이퍼파라미터 및 실행 설정
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Cora', help='Dataset name (Cora, Citeseer, Pubmed)')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--w_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--hidden_channels', type=int, default=16, help='Hidden layer dimension')
parser.add_argument('--device', type=int, default=0, help='GPU device ID (if available)')
args = parser.parse_args()
# config = yaml.load(open(args.config, encoding="utf-8"), Loader=SafeLoader)[args.dataset]

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

embed_dim = 128
cancerType = 'pan-cancer'

def extract_edge_data_with_score(sparse_tensor):
    sparse_tensor = sparse_tensor.coalesce()
    row, col = sparse_tensor.indices()
    score = sparse_tensor.values()
    return row, col, score

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        loss = self.alpha * ((1 - pt) ** self.gamma) * BCE
        return loss.mean()


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



# 2. 데이터 입력
dataPath = "./Data/CPDB"
# load new multi-omics feature 
data_x_df = pd.read_csv(dataPath + '/multiomics_features_SNV_METH_GE_CNA_filtered_CPDB.tsv', sep='\t', index_col=0)
data_x_df = data_x_df.dropna()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_x_df.values)
data_x = torch.tensor(features_scaled, dtype=torch.float32, device=device)
data_x = data_x[:,:48]

if cancerType=='pan-cancer':
    data_x = data_x[:,:48]
else:
    cancerType_dict = {
                        'kirc':[0,16,32],
                        'brca':[1,17,33],
                        'prad':[3,19,35],
                        'stad':[4,20,36],
                        'hnsc':[5,21,37],
                        'luad':[6,22,38],
                        'thca':[7,23,39],
                        'blca':[8,24,40],
                        'esca':[9,25,41],
                        'lihc':[10,26,42],
                        'ucec':[11,27,43],
                        'coad':[12,28,44],
                        'lusc':[13,29,45],
                        'cesc':[14,30,46],
                        'kirp':[15,31,47]
                                }
    data_x = data_x[:, cancerType_dict[cancerType]]

ppiAdj = torch.load(dataPath+'/CPDB_ppi.pkl')
pathAdj = torch.load(dataPath+'/pathway_SimMatrix_filtered.pkl')
goAdj = torch.load(dataPath+'/GO_SimMatrix_filtered.pkl')

msigdb_genelist = pd.read_csv('./Data/msigdb/geneList.csv', header=None)
msigdb_genelist = list(msigdb_genelist[0].values)
string_ppi_path = f'{dataPath}/CPDB_ppi_edgelist.tsv'
incidenceMatrix = processingIncidenceMatrix(msigdb_genelist, string_ppi_path, dataPath)

node_features = data_x  # torch.Tensor, [N, 48]
gene_set_matrix = torch.tensor(incidenceMatrix.values, dtype=torch.float32, device=device)


ppi_row, ppi_col, ppi_score = extract_edge_data_with_score(ppiAdj)
path_row, path_col, path_score = extract_edge_data_with_score(pathAdj)
go_row, go_col, go_score = extract_edge_data_with_score(goAdj)

# # edge_indices 구성
# edge_indices = {
#     0: (ppiAdj.coalesce().indices()[0], ppiAdj.coalesce().indices()[1]),     # STRING
#     1: (pathAdj.coalesce().indices()[0], pathAdj.coalesce().indices()[1]),   # KEGG Pathway
#     2: (goAdj.coalesce().indices()[0], goAdj.coalesce().indices()[1])        # GO Similarity
# }

edge_indices_with_score = {
    "ppi": (ppi_row, ppi_col, ppi_score),     # STRING with confidence
    "path": (path_row, path_col, path_score),  # Pathway with similarity
    "go": (go_row, go_col, go_score)         # GO with similarity
}
    
edge_index_dict = {
    'ppi': torch.stack([ppi_row, ppi_col], dim=0).to(device),     # [2, num_edges]
    'path': torch.stack([path_row, path_col], dim=0).to(device),
    'go': torch.stack([go_row, go_col], dim=0).to(device),
}

# Random Walk PE 및 PageRank 중심성 계산
torch_ppi_dense = ppiAdj.to_dense().cpu().numpy()
ppi_sp = sp.csr_matrix(torch_ppi_dense)
G = nx.from_scipy_sparse_matrix(ppi_sp)

# 1. PageRank 중심성
pagerank_dict = nx.pagerank(G, alpha=0.85)
pagerank_vec = torch.tensor([pagerank_dict[i] for i in range(len(pagerank_dict))], dtype=torch.float32, device=device).unsqueeze(1)  # [N, 1]

# # 2. Random Walk PE
# rw_mat = ppi_sp.astype(np.float32)
# rw_mat = rw_mat.multiply(1.0 / rw_mat.sum(axis=1).A)
# rw_pe = torch.tensor(rw_mat.toarray(), dtype=torch.float32, device=device)  # [N, N] or truncate to [N, k]

# 2. Random Walk PE
deg_row = ppi_sp.sum(axis=1).A1
deg_row[deg_row == 0] = 1.0  # divide-by-zero 방지
rw_mat = ppi_sp.multiply(1.0 / deg_row[:, None])
rw_pe = torch.tensor(rw_mat.toarray(), dtype=torch.float32, device=device)[:, :embed_dim]  # [N, embed_dim]

# 3. SPD bias
# dist_matrix = shortest_path(ppi_sp, directed=False, unweighted=True)
# dist_matrix[np.isinf(dist_matrix)] = dist_matrix[~np.isinf(dist_matrix)].max() + 1
# sigma = 1.0
# spd_bias = -torch.tensor(np.exp(-(dist_matrix ** 2) / (2 * sigma ** 2)), dtype=torch.float32, device=device)  # [N, N]

# # 3. 최대 거리 클리핑 (bucket 수 제한)
# dist_matrix = shortest_path(ppi_sp, directed=False, unweighted=True)
# dist_matrix[np.isinf(dist_matrix)] = dist_matrix[~np.isinf(dist_matrix)].max() + 1
# max_dist = 10
# spd_bucket = np.clip(dist_matrix, 0, max_dist).astype(np.int64)  # [N, N] integer matrix
# spd_bucket = torch.tensor(spd_bucket, device=device)

cross_val=10    
AUC = np.zeros(shape=(cross_val))
AUPR = np.zeros(shape=(cross_val))
F1_SCORES = np.zeros(cross_val)

print("----- pan-cancer 10-fold training start ------")
for i in range(cross_val):
    print(f'-------- Fold {i+1} Begin ---------')
    fold_path = f"{dataPath}/CPDB_10fold/fold_{i+1}"
    train_idx, valid_idx, test_idx, train_mask, valid_mask, test_mask, Y = load_kfold_data(fold_path, device)

    model = Graphomer(
        input_dim=node_features.shape[1],
        gene_set_dim=gene_set_matrix.shape[1],
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = FocalLoss(alpha=1.0, gamma=1.5)
    
    best_val_auc = 0.0
    best_epoch = -1
    best_model_state = None
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, rw_pe, pagerank_vec)  # shape: [N]
        loss = loss_fn(logits[train_idx], Y[train_idx])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred_logits = model(node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, rw_pe, pagerank_vec)
            pred_probs = torch.sigmoid(pred_logits)

            # validation performance
            val_auc = roc_auc_score(Y[valid_idx].cpu(), pred_probs[valid_idx].cpu())
            val_aupr = average_precision_score(Y[valid_idx].cpu(), pred_probs[valid_idx].cpu())
            print(f"[Fold {i+1}] Epoch {epoch+1} | Val AUC: {val_auc:.4f}, AUPR: {val_aupr:.4f}")

            # if val_auc > best_val_auc:
            #     best_val_auc = val_auc
            #     best_epoch = epoch+1
            #     best_model_state = model.state_dict()  # 모델 파라미터 저장
        # if epoch % 5 == 0 or epoch == args.epochs - 1:
        #     model.eval()
        #     with torch.no_grad():
        #         pred_logits = model(node_features, gene_set_matrix, edge_indices)
        #         pred_probs = torch.sigmoid(pred_logits)

        #         # validation performance
        #         val_auc = roc_auc_score(Y[valid_idx].cpu(), pred_probs[valid_idx].cpu())
        #         val_aupr = average_precision_score(Y[valid_idx].cpu(), pred_probs[valid_idx].cpu())
        #         print(f"[Fold {i+1}] Epoch {epoch} | Val AUC: {val_auc:.4f}, AUPR: {val_aupr:.4f}")
    
    # print best_epoch
    # print(f'------ Fold {i+1} Best Epoch : {best_epoch} ------')
    
    # Final evaluation on test set
    # model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, rw_pe, pagerank_vec)
        final_probs = torch.sigmoid(final_logits)
        pred_labels = (final_probs > 0.5).float()

        AUC[i] = roc_auc_score(Y[test_idx].cpu(), final_probs[test_idx].cpu())
        AUPR[i] = average_precision_score(Y[test_idx].cpu(), final_probs[test_idx].cpu())
        F1_SCORES[i] = f1_score(Y[test_idx].cpu(), pred_labels[test_idx].cpu())

        # ➤ 독립 평가용 (OncoKB, ONGene)
        all_probs = final_probs.cpu().detach()

        # ➤ 유전자 이름 ↔ 인덱스 매핑
        feature_genes = pd.read_csv("./Data/CPDB/feature_genename.txt", header=None)[0].tolist()
        gene_to_index = {gene: idx for idx, gene in enumerate(feature_genes)}
        train_gene_set = set(train_idx.tolist())

        def get_independent_test_indices(path):
            genes = pd.read_csv(path, header=None)[0].tolist()
            return [gene_to_index[g] for g in genes if g in gene_to_index and gene_to_index[g] not in train_gene_set]

        def compute_auprc(target_indices, pred_probs):
            label_tensor = torch.zeros_like(pred_probs)
            label_tensor[target_indices] = 1
            return average_precision_score(label_tensor.numpy(), pred_probs.numpy())

         # ➤ Recall 계산 함수
        def compute_recall(target_indices, pred_labels):
            label_tensor = torch.zeros_like(pred_labels)
            label_tensor[target_indices] = 1
            return recall_score(label_tensor.numpy(), pred_labels.numpy())

        # ➤ OncoKB / ONGene AUPRC
        onco_test_idx = get_independent_test_indices("./Data/OncoKB_ONGene/OncoKB_CPDB.txt")
        ongene_test_idx = get_independent_test_indices("./Data/OncoKB_ONGene/ONGene_CPDB.txt")

        onco_auprc = compute_auprc(onco_test_idx, all_probs)
        ongene_auprc = compute_auprc(ongene_test_idx, all_probs)

        # ➤ Recall 계산
        onco_recall = compute_recall(onco_test_idx, pred_labels.cpu())
        ongene_recall = compute_recall(ongene_test_idx, pred_labels.cpu())

        os.makedirs('./independent_test_cpdb', exist_ok=True)
        with open(f"./independent_test_cpdb/auprc_independent_fold_{i+1}_cpdb.txt", "w") as f:
            f.write(f"OncoKB_AUPRC\t{onco_auprc:.4f}\n")
            f.write(f"ONGene_AUPRC\t{ongene_auprc:.4f}\n")
            f.write(f"OncoKB_Recall\t{onco_recall:.4f}\n")
            f.write(f"ONGene_Recall\t{ongene_recall:.4f}\n")

        # 유전자 이름 로딩 (index 순서대로)
        with open('./Data/CPDB/feature_genename.txt') as f:
            gene_names = [line.strip() for line in f.readlines()]
        
            assert len(gene_names) == final_probs.shape[0], "유전자 이름 개수와 예측 결과 개수가 일치하지 않습니다."

        # CPU로 이동하여 numpy로 변환
        pred_probs_all = final_probs.cpu().numpy()
        true_labels_all = Y.cpu().numpy()

        # # 유전자 이름, 예측 확률, 라벨 결합
        # gene_preds = np.column_stack((gene_names, pred_probs_all, true_labels_all))

        # # numpy 배열로 명시적으로 분리
        # gene_names_arr = np.array(gene_names, dtype=str).reshape(-1, 1)
        # pred_probs_arr = final_probs.cpu().numpy().reshape(-1, 1).astype(float)
        # true_labels_arr = Y.cpu().numpy().reshape(-1, 1).astype(float)

        # # object 타입 배열로 결합
        # gene_preds = np.hstack([gene_names_arr, pred_probs_arr, true_labels_arr])
        # numpy 배열로 각각 준비
        gene_names_arr = np.array(gene_names, dtype=object).reshape(-1, 1)
        pred_probs_arr = final_probs.cpu().numpy().reshape(-1, 1).astype(float)
        true_labels_arr = Y.cpu().numpy().reshape(-1, 1).astype(float)

        # object 배열로 결합
        gene_preds = np.hstack((gene_names_arr, pred_probs_arr, true_labels_arr)).astype(object)
        

        # prediction_genes_STRING 폴더 없으면 생성
        os.makedirs('./prediction_genes_CPDB', exist_ok=True)

        # 저장
        np.savetxt(
            f'./prediction_genes_CPDB/predictions_fold_{i+1}.txt',
            gene_preds,
            fmt='%s\t%.7f\t%.1f',
            delimiter='\t',
            header='Gene\tPredicted\tLabel',
            comments=''
        )
        
    print(f"Fold {i+1} Results — AUC: {AUC[i]:.3f}, AUPR: {AUPR[i]:.3f}, F1: {F1_SCORES[i]:.3f}")

    # 각 fold 결과를 텍스트 파일에 기록
    with open("./ablation_model11_nonattnbias_final_results.txt", "a") as result_file:
        result_file.write(f"Fold {i+1}: AUC={AUC[i]:.3f}, AUPR={AUPR[i]:.3f}, F1-score={F1_SCORES[i]:.3f}\n")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
print("========== Final 10-Fold Results ==========")
print(f"Mean AUC: {AUC.mean():.3f} ± {AUC.std():.3f}")
print(f"Mean AUPR: {AUPR.mean():.3f} ± {AUPR.std():.3f}")
print(f"Mean F1: {F1_SCORES.mean():.3f} ± {F1_SCORES.std():.3f}")

with open("./ablation_model11_nonattnbias_final_results.txt", "a") as result_file:
    result_file.write(f"\nFinal Results:\nMean AUC: {AUC.mean():.3f} ± {AUC.std():.3f}\nMean AUPR: {AUPR.mean():.3f} ± {AUPR.std():.3f}\nMean F1-score: {F1_SCORES.mean():.3f} ± {F1_SCORES.std():.3f}\n")

