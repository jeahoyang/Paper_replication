# -*- coding: utf-8 -*-      
import torch
import numpy as np
import time 
from models.granger_model import GRANGER, train_phase1

device = torch.device('cuda')
X_np = np.load('example_data/mCAD-2000-1/time_output.npy').T 
# print(X_np.shape) 
dim = X_np.shape[-1] 
GC = np.zeros([dim,dim])
for i in range(dim):
    GC[i,i] = 1
    if i!=0:
        GC[i,i-1] = 1
X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
smaple=X.shape[0]
full_connect = np.ones(GC.shape)
granger = GRANGER(X_np,X.shape[-1],full_connect, hidden=256).cuda(device=device)

start_time = time.time() 
train_loss_list = train_phase1(
    granger, X, context=20, lam=0.3, lam_ridge=0, lr=3e-3, max_iter=500, 
    check_every=50,batch_size=8)#0.1
end_time = time.time() 
print(f"Total time taken: {end_time - start_time} seconds")