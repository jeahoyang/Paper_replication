import argparse
import numpy as np
import torch
from sklearn import metrics
from model import ECD_CDGINet
from model_all import Net
from data.data_loader import load_net_specific_data
import torch.nn.functional as F

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score, auc

parser = argparse.ArgumentParser()
parser.add_argument('--is_5_CV_test', type=bool, default=True, help='Run 5-CV test.')
parser.add_argument('--dataset_file', type=str, default='./data/PathNet/dataset_PathNet_ten_5CV.pkl',
                    help='The path of the input pkl file.')  # When setting is_5_CV_test=True, make sure the pkl file include masks of different 5CV splits.
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--w_decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--in_channels', type=int, default=58, help='Dimension of node features.')
parser.add_argument('--in_channels', type=int, default=48, help='Dimension of node features.')
parser.add_argument('--hidden_channels', type=int, default=100, help='Dimension of hidden Linear layers.')
parser.add_argument('--device', type=int, default=0, help='The id of GPU.')
args = parser.parse_args()
device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
data = load_net_specific_data(args)
data = data.to(device)

print(type(data))
print(data)

print(data.x)
print(data.edge_index)
print(data.y)

print('new data check ----------------')

# new data path
dataPath = "./new_data/string/"

# load new multi-omics feature  
data_x_df = pd.read_csv(dataPath + 'multiomics_features_SNV_METH_GE_CNA_filtered_string.tsv', sep='\t', index_col=0)
data_x_df = data_x_df.dropna()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_x_df.values)
data_x = torch.tensor(features_scaled, dtype=torch.float32, device=device)
data_x = data_x[:,:48]
print(data_x)

# load new edgelist data
ppiAdj = torch.load(dataPath + 'string_ppi.pkl')
ppiAdj_index = ppiAdj.coalesce().indices().to(device)
print(ppiAdj_index)

# load new label data
# labels = torch.tensor(np.loadtxt(dataPath + '10fold/fold_1/labels.txt'), dtype=torch.float32, device=device)
# print(labels)

# Prepare result storage
AUC_scores = []
AUPR_scores = []
F1_scores = []


# @torch.no_grad()
# def test(data, mask):
#     model.eval()
#     x = model(data_x, ppiAdj_index)
#     pred = torch.sigmoid(x[mask])
#     precision, recall, _thresholds = metrics.precision_recall_curve(data.y[mask].cpu().numpy(),
#                                                                     pred.cpu().detach().numpy())
#     area = metrics.auc(recall, precision)
#     return metrics.roc_auc_score(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy()), area, data.y[
#         mask].cpu().numpy(), pred.cpu().detach().numpy()


# Evaluation function
@torch.no_grad()
def test(model, mask, labels, fold_idx, gene_names, train_idx):
    model.eval()
    x = model(data_x, ppiAdj_index)
    pred = torch.sigmoid(x[mask])
    # Pred / Yn 저장
    Yn = labels[mask]
    
    # 유전자 인덱스 또는 유전자 이름과 예측값 저장
    gene_info = np.array(gene_names)[mask.cpu().numpy()]
    
    
    # Move pred and Yn to CPU for further processing
    pred_cpu = pred.cpu().detach().numpy()
    Yn_cpu = labels[mask].cpu().numpy()
    
    gene_predictions = np.column_stack((gene_info, pred_cpu, Yn_cpu))
    np.savetxt(f"./prediction_genes_string/predictions_fold_{fold_idx}.txt", gene_predictions, fmt="%s", delimiter="\t", header="Gene\tPredicted\tLabel", comments="")
    
    # precision, recall, _ = metrics.precision_recall_curve(labels[mask].cpu().numpy(), pred.cpu().detach().numpy())
    auc_roc = metrics.roc_auc_score(labels[mask].cpu().numpy(), pred.cpu().detach().numpy())
    auprc = average_precision_score(Yn_cpu, pred_cpu)
    f1_score = metrics.f1_score(Yn_cpu, (pred_cpu > 0.5).astype(int))
    
    # ➤ 독립 평가용 (OncoKB, ONGene)
    all_probs = torch.sigmoid(x).cpu().detach()

    # 여기 코드를 내 코드 인덱스 맞게 매칭 시키고 ㄱㄱㄱㄱ
    # ➤ 유전자 이름 ↔ 인덱스 매핑
    feature_genes = pd.read_csv("./new_data/string/feature_genename.txt", header=None)[0].tolist()
    gene_to_index = {gene: idx for idx, gene in enumerate(feature_genes)}
    train_gene_set = set(train_idx.tolist())

    def get_independent_test_indices(path):
        genes = pd.read_csv(path, header=None)[0].tolist()
        return [gene_to_index[g] for g in genes if g in gene_to_index and gene_to_index[g] not in train_gene_set]

    def compute_auprc(target_indices, pred_probs):
        label_tensor = torch.zeros_like(pred_probs)
        label_tensor[target_indices] = 1
        return average_precision_score(label_tensor.numpy(), pred_probs.numpy())

    # ➤ OncoKB / ONGene AUPRC
    onco_test_idx = get_independent_test_indices("./new_data/OncoKB_ONGene/OncoKB_string.txt")
    ongene_test_idx = get_independent_test_indices("./new_data/OncoKB_ONGene/ONGene_string.txt")

    onco_auprc = compute_auprc(onco_test_idx, all_probs)
    ongene_auprc = compute_auprc(ongene_test_idx, all_probs)

    # ➤ 로그 출력 및 저장
    print(f"✅ Fold {fold_idx} | AUROC: {auc_roc:.3f} | AUPRC: {auprc:.3f} | F1: {f1_score:.3f}")
    print(f"🔬 OncoKB AUPRC: {onco_auprc:.3f} | ONGene AUPRC: {ongene_auprc:.3f}")

    with open(f"./independent_test_string/auprc_independent_fold_{fold_idx}_string.txt", "w") as f:
        f.write(f"OncoKB_AUPRC\t{onco_auprc:.4f}\n")
        f.write(f"ONGene_AUPRC\t{ongene_auprc:.4f}\n")
    
    
    return auc_roc, auprc, f1_score

# # Ten times of 5_CV
# AUC = np.zeros(shape=(10, 5))
# AUPR = np.zeros(shape=(10, 5))

# for i in range(10):
#     for cv_run in range(5):
#         tr_mask, te_mask = data.mask[i][cv_run]
#         model = ECD_CDGINet(args).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#         for epoch in range(1, args.epochs + 1):
#             # Training model
#             model.train()
#             optimizer.zero_grad()
#             pred = model(data_x, ppiAdj_index)
#             loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1))
#             loss.backward()
#             optimizer.step()
#         AUC[i][cv_run], AUPR[i][cv_run], a, b = test(data, te_mask)
#         a, b, all_y_true1, all_y_scores1 = test(data, te_mask)
#         print(f'Training epoch: {epoch:03d}')
#         # print('Round--%d CV--%d  AUC: %.5f, AUPR: %.5f' % (i, cv_run + 1, AUC[i][cv_run], AUPR[i][cv_run]))
#         print('Round--%d CV--%d  AUC: %.5f, AUPR: %.5f' % (i, cv_run + 1, AUC[i][cv_run], AUPR[i][cv_run]))
#     print('Round--%d Mean AUC: %.5f, Mean AUPR: %.5f' % (i, np.mean(AUC[i, :]), np.mean(AUPR[i, :])))

# print('10 rounds for 5CV-- Mean AUC: %.4f, Mean AUPR: %.4f' % (AUC.mean(), AUPR.mean()))
feature_genename_file = './feature_genename.txt'  # feature_genename.txt 파일 경로
geneList = pd.read_csv(feature_genename_file, header=None).iloc[:, 0].tolist()

# Loop over 10-fold cross-validation
for fold in range(1, 11):
    # fold_path = dataPath + f'10fold/fold_{fold}/'
    fold_path = dataPath + f'10fold/fold_{fold}/'
    

    # Load masks and labels
    train_mask = torch.tensor(np.loadtxt(f"{fold_path}/train_mask.txt"), dtype=torch.bool, device=device)
    valid_mask = torch.tensor(np.loadtxt(f"{fold_path}/valid_mask.txt"), dtype=torch.bool, device=device)
    test_mask = torch.tensor(np.loadtxt(f"{fold_path}/test_mask.txt"), dtype=torch.bool, device=device)
    labels = torch.tensor(np.loadtxt(f"{fold_path}/labels.txt"), dtype=torch.float32, device=device)

    train_idx = torch.tensor(np.loadtxt(f"{fold_path}/train.txt"), dtype=torch.long, device=device)
    
    # Model initialization
    model = ECD_CDGINet(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(data_x, ppiAdj_index)
        loss = F.binary_cross_entropy_with_logits(pred[train_mask], labels[train_mask].view(-1, 1))
        loss.backward()
        optimizer.step()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

    # Evaluation on test set
    auc_roc, auc_pr, f1_score = test(model, test_mask, labels, fold, geneList, train_idx)
    AUC_scores.append(auc_roc)
    AUPR_scores.append(auc_pr)
    F1_scores.append(f1_score)

    with open("./final_results_string.txt", "a") as result_file:
        result_file.write(f"Fold {fold}: AUC={AUC_scores[fold-1]:.3f}, AUPR={AUPR_scores[fold-1]:.3f}, F1-score={F1_scores[fold-1]:.3f}\n")
    # # Save results
    # with open(f"{args.dataset_path}/fold_{fold}/evaluation_results.txt", "w") as f:
    #     f.write(f"Fold {fold} - AUROC: {auc_roc:.5f}, AUPRC: {auc_pr:.5f}\n")

    print(f"Fold {fold}: AUROC = {auc_roc:.3f}, AUPRC = {auc_pr:.3f}, F1-score = {f1_score:.3f}")

# Calculate and print mean scores
mean_auc = np.mean(AUC_scores)
std_auc = np.std(AUC_scores)

mean_aupr = np.mean(AUPR_scores)
std_aupr = np.std(AUPR_scores)

mean_f1_score = np.mean(F1_scores)
std_f1 = np.std(F1_scores)

# Print mean and std
print(f"Mean AUC: {mean_auc:.3f}, Std AUC: {std_auc:.3f}")
print(f"Mean AUPR: {mean_aupr:.3f}, Std AUPR: {std_aupr:.3f}")
print(f"Mean F1-score: {mean_f1_score:.3f}, Std F1-score: {std_f1:.3f}")


# Save overall results
with open(f"./overall_evaluation_string.txt", "w") as f:
    f.write(f"Mean AUROC: {mean_auc:.3f}, Mean AUPRC: {mean_aupr:.3f}, Mean F1-score: {mean_f1_score:.3f}\n")