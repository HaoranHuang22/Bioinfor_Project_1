from torch import optim
from torch.optim import Adam
from tqdm import tqdm

from utils.data import read_domain_ids_per_chain_from_txt
from utils.dataset import *
from utils.plot import *
from diffusion_model.embed import *
from diffusion_model.structure_diffusion_model import *
from loss.FAPE import *

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 20
    LEARNING_RATE = 1e-3
    fixed_length = 128

    # Load data
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

    diffusion = ProteinDiffusion(device=DEVICE)
    model = StructureModel().to(DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr = 1e-3)

    # Train
    for epoch in range(EPOCH):    
        model.train()
        train_loss = 0
        for batch_idx, (pdb, res_label, atom_coords, mask) in enumerate(tqdm(train_loader, leave=False)):
            # Data preparation
            atom_coords = atom_coords.to(torch.float32)
            n_coords = atom_coords[:, :, 0]
            ca_coords = atom_coords[:, :, 1]
            c_coords = atom_coords[:, :, 2]
            rotations, translations = rigidFrom3Points(n_coords, ca_coords, c_coords)
            q_0 = roma.rotmat_to_unitquat(rotations.mT) # column vectors
            single_repr = get_single_representation(pdb, res_label).to(DEVICE)
            pair_repr = torch.cdist(ca_coords, ca_coords, p=2).to(torch.float32)
            
            # Idealized atom coordinates
            n_coords_ideal = torch.Tensor([-0.525, 1.363, 0.0]).expand(n_coords.shape)
            ca_coords_ideal = torch.Tensor([0.0, 0.0, 0.0]).expand(ca_coords.shape)
            c_coords_ideal = torch.Tensor([1.526, 0.0, 0.0]).expand(c_coords.shape)

            # Foward Diffusion
            batch_size = atom_coords.shape[0]
            time = diffusion.sample_timesteps(batch_size = batch_size)
            x_t = diffusion.coord_q_sample(ca_coords, time).to(torch.float32)
            q_t = diffusion.quaternion_q_sample(q_0, time)

            # train model
            quaternions_pred, translations_pred, n_coords_pred, c_coords_pred = model(single_repr, pair_repr.to(device=DEVICE), q_t.to(device=DEVICE), x_t.to(device=DEVICE))
            optimizer.zero_grad()
            loss = (computeFAPE(quaternions_pred, translations_pred, translations_pred, q_0.to(device=DEVICE), translations.to(device=DEVICE), ca_coords_ideal.to(device=DEVICE)) + 
                    computeFAPE(quaternions_pred, translations_pred, n_coords_pred, q_0.to(device=DEVICE), translations.to(device=DEVICE), n_coords_ideal.to(device=DEVICE)) +
                    computeFAPE(quaternions_pred, translations_pred, c_coords_pred, q_0.to(device=DEVICE), translations.to(device=DEVICE), c_coords_ideal.to(device=DEVICE))) / 3

            loss.backward() # calc gradients
            train_loss += loss.item()
            optimizer.step() # backpropagation
        print('====> Epoch: {}  Total loss: {:.10f}'.format(epoch, train_loss / len(train_loader.dataset)))

    # Save model
    torch.save(model.state_dict(), './model/structure_model_128.pt')

