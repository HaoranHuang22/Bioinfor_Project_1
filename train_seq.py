import torch
from torch import optim
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import repeat

import pandas as pd
from scipy.stats import binom

from utils.data import *
from common.res_infor import *
from utils.dataset import *
from diffusion_model.sequence_diffusion_model import *

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 100
    LEARNING_RATE = 1e-3
    fixed_length = 128

    # load data
    train_data = pd.read_csv('./data/train_data.csv')
    test_data = pd.read_csv('./data/test_data.csv')

    # subsample data
    train_data = subsample_protein_data(train_data, fixed_length=fixed_length)
    test_data = subsample_protein_data(test_data, fixed_length=fixed_length)

    # Data loader
    train_dataset = ProteinDataset(train_data)
    test_dataset = ProteinDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128)

    # Model
    diffusion = SequenceDiffusion(res_num=fixed_length, device=DEVICE)
    model = SequenceModel().to(DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr = 1e-3)

    # Train
    loss_history = []
    for epoch in range(EPOCH):    
        model.train()
        train_loss = 0
        for batch_idx, (pdb, res_label, atom_coords, mask) in enumerate(tqdm(train_loader, leave=False)):
            # Data preparation
            x_0 = res_label.to(DEVICE)
            x_0 = x_0.to(torch.float32)
        
            n_coords = atom_coords[:, :, 0]
            ca_coords = atom_coords[:, :, 1]
            c_coords = atom_coords[:, :, 2]

            pair_repr = torch.cdist(ca_coords, ca_coords, p=2)
            rotations, translations = rigidFrom3Points(n_coords, ca_coords, c_coords)
            
            # Foward Diffusion
            batch_size = atom_coords.shape[0]
            t, pt = diffusion.sample_timesteps(batch_size = batch_size, device=DEVICE)

            x_t, x_0_ignore, mask = diffusion.seq_q_sample(x_0, t)
            
            # Backward Diffusion
            x_0_hat_logits = model(x_t, pair_repr.to(DEVICE, torch.float32), rotations.to(DEVICE,torch.float32), translations.to(DEVICE, torch.float32))
            
            
            
            # Custom loss function
            cross_entropy_loss = F.cross_entropy(x_0_hat_logits.transpose(1, 2), 
                                                x_0_ignore.to(torch.int64), 
                                                ignore_index = -1, reduction='none')
            
            cross_entropy_loss = torch.sum(cross_entropy_loss, dim=1)

            vb_loss = cross_entropy_loss / t

            vb_loss = vb_loss / pt
            
            loss = vb_loss.mean()

            optimizer.zero_grad()
            loss.backward() # calc gradients
            train_loss += loss.item()
            optimizer.step() # backpropagation

        avg_loss = train_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, avg_loss))

    # Save model
    torch.save(model.state_dict(), './trained_models/sequence_model_128.pt')

