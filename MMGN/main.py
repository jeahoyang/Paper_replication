import torch
from utils.dgmimodel import DMGI
from utils.datahandle import DataIO
import argparse
import time
from indentify_cancer_gene import deepod_dsvdd
from embedding import train_embedding, get_embedding
import random
import numpy as np
from torch_geometric.data import DataLoader

# 랜덤 시드 설정
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# ArgumentParser 설정
parser = argparse.ArgumentParser()
parser.add_argument('--network_paths',
                    default=['PPI=./processed_dataset/six_net/PPI_for_pyG.txt',
                             'Pathway=./processed_dataset/six_net/Pathway_for_pyG.txt',
                             'Kinase=./processed_dataset/six_net/Kinase_for_pyG.txt',
                             'Metabolic=./processed_dataset/six_net/Metabolic_for_pyG.txt',
                             'Regulatory=./processed_dataset/six_net/Regulatory_for_pyG.txt',
                             'Complexes=./processed_dataset/six_net/Complexes_for_pyG.txt'],
                    nargs='+',
                    metavar='KEY=VALUE',
                    help='dictionary of network paths')
parser.add_argument('--feature_path',
                    default='./processed_dataset/six_net/Feature_for_pyG.csv',
                    help='Dictionary of feature path')
parser.add_argument('--mapping_dict',
                    default='./processed_dataset/six_net/mapping_dict_EIDtoIndex.pickle',
                    help='Dictionary of mapping dict path')
parser.add_argument("--kcg", default='./processed_dataset/six_net/kcg_intersec.csv', help='Know cancer gene')
parser.add_argument('--embedding_file_path', default="./saved_embedding/six_net", help='the file path of embedding')
parser.add_argument('--dimension', type=int, default=256, help='the dimension of embedding')
parser.add_argument('--lr_dmgi', type=float, default=0.001, help='learning rate of train')
parser.add_argument('--wd_dmgi', type=float, default=0.01, help='weight decay of train')

args = parser.parse_args()

# 네트워크 경로 딕셔너리 변환
network_paths = {item.split('=')[0]: item.split('=')[1] for item in args.network_paths}

# 데이터 로드
dataIO = DataIO()
data, net_types = dataIO.load_network(
    Features_path=args.feature_path, need_features=True, **network_paths)  
EID_to_index = dataIO.load_mapping_dict(args.mapping_dict)
kcg_intersec = dataIO.load_know_cancer_gene(args.kcg)

node_types, edge_types = data.metadata()
edge_types_num = len(edge_types)

# DMGI 모델 설정
model_DMGI = DMGI(data['gene'].num_nodes, data['gene'].x.size(-1),
                  out_channels=args.dimension,
                  num_relations=edge_types_num)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model_DMGI = data.to(device), model_DMGI.to(device)
optimizer_DMGI = torch.optim.RMSprop(model_DMGI.parameters(), lr=args.lr_dmgi, weight_decay=args.wd_dmgi)

if __name__ == '__main__':
    for fold in range(1, 11):  # fold_1 ~ fold_10
        print(f"\n========== Processing Fold {fold} ==========")

        train_file = f"STRING_10foldTEST/fold_{fold}/train.txt"
        test_file = f"STRING_10foldTEST/fold_{fold}/test.txt"

        Maybe_cancergene_train = dataIO.load_maybe_cancer_gene(train_file)
        Maybe_cancergene_test = dataIO.load_maybe_cancer_gene(test_file)

        ### Embedding 학습
        print(f"Training embedding for fold {fold}...")
        train_embedding(model_DMGI, optimizer_DMGI, data, net_types)

        print(f"Getting embedding for fold {fold}...")
        embedding = get_embedding(model_DMGI)
        encoded_embedding = embedding
        
        print('getting embeddddddddddddddddddddddddding')
        
        
        # ### 결과 저장
        # embedding_save_path = f"./saved_embedding/six_net_fold_{fold}.npy"
        # np.save(embedding_save_path, encoded_embedding)
        # print(f"Saved embedding to {embedding_save_path}")

        ### Deep SVDD 실행
        print(f"Running Deep SVDD for fold {fold}...")
        cancer_genes_proba = deepod_dsvdd(kcg_intersec, encoded_embedding, EID_to_index, Maybe_cancergene_test)

        print(f"Fold {fold} completed!\n")
