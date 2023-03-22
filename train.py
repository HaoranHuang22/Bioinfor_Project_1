from torch import optim
from torch.optim import Adam
from tqdm import tqdm

from utils.data import read_domain_ids_per_chain_from_txt
from utils.dataset import *
from diffusion_model.embed import *
from diffusion_model.structure_diffusion_model import *

if __name__ == "__main__":

    #global setting
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 100
    LEARNING_RATE = 1e-3

    #load data
    train_pdbs, train_pdb_chains = read_domain_ids_per_chain_from_txt('./data/train_domains.txt')
    test_pdbs, test_pdb_chains = read_domain_ids_per_chain_from_txt('./data/test_domains.txt')
    train_loader = BackboneCoordsDataLoader(train_pdb_chains, "./data/train_backbone_coords_20.npy", "./data/train_data_res_20.npy",seq_length=20, batch_size=128, shuffle=True)
    test_loader = BackboneCoordsDataLoader(test_pdb_chains, './data/test_backbone_coords_20.npy', './data/test_data_res_20.npy', seq_length=20, batch_size=128, shuffle=True)

    #model define
    diffusion = ProteinDiffusion(device=DEVICE)
    model = StructureModel().to(DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr = 1e-3)

    #training
    for epoch in range(EPOCH):    
        model.train()
        train_loss = 0
        for batch_idx, (pdb, res_label, atom_coords) in enumerate(tqdm(train_loader, leave=False)):
            # Data preparation
            atom_coords = atom_coords.to(DEVICE)
            atom_coords = atom_coords.to(torch.float32)
            n_coords = atom_coords[:, :, 0]
            ca_coords = atom_coords[:, :, 1]
            c_coords = atom_coords[:, :, 2]
            R, t = rigidFrom3Points(n_coords, ca_coords, c_coords)
            q_0 = roma.rotmat_to_unitquat(R)
            single_repr = get_single_representation(pdb, res_label).to(DEVICE)
            
            pair_repr = torch.cdist(ca_coords, ca_coords, p=2).to(torch.float32)
            

            # Foward Diffusion
            batch_size = atom_coords.shape[0]
            t = diffusion.sample_timesteps(batch_size = batch_size).to(DEVICE)
            x_t = diffusion.coord_q_sample(ca_coords, t).to(torch.float32)
            q_t = diffusion.quaternion_q_sample(q_0, t)

            # train model
            pred_coords = model(single_repr, pair_repr, q_t, x_t)
            optimizer.zero_grad()
            loss = F.mse_loss(pred_coords, ca_coords)
            loss.backward() # calc gradients
            train_loss += loss.item()
            optimizer.step() # backpropagation
        print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / len(train_loader.dataset)))