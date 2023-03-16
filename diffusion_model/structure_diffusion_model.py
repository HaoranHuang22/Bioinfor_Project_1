import torch 
import torch.nn as nn
import torch.nn.functional as F

from common.res_infor import label_res_dict

import esm
# Constraint Part

def get_single_representation(pdb_chain, res_label):
    """
    Function to get single representation

    Args:
        pdb_chain(tuple): batch_size
        res_label(torch.Tensor): dim -> (batch_size, num_res, 1)

    Returns:
        single_repr(torch.Tensor): dim -> (batch_size, embedding_dim)
    """
    #Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval() # disables dropout for deterministic results

    #Prepare data
    data = []
    for i in range(len(pdb_chain)):
        sequence = ""
        for label in res_label[i]:
            aa = label_res_dict[label.item()]
            sequence += aa
        data.append((pdb_chain[i], sequence))

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    
    single_repr = torch.stack(sequence_representations, dim=0)
    return single_repr

def get_secondary_representation(atom_coords: torch.Tensor):
    """
    Function to get pair representation, C alpha atom distance map

    Args:
        atom_coords(torch.Tensor): dim -> (batch_size, num_res, 3, 3)

    Returns:
        pair_repr(torch.Tensor): dim -> (batch_size, num_res, num_res)
    """
    ca_coords = atom_coords[:,:,1]

    # Compute the distance matrix
    dist_mat = torch.cdist(ca_coords, ca_coords, p=2)
    return dist_mat

# Diffusion Part

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original paper
    https://arxiv.org/abs/2006.11239
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    https://openreview.net/forum?id=-NEXDKk8gZ
    """
    t_T = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float64) / timesteps
    f_t = torch.cos((t_T + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class Diffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_schedule = 'cosine', device = None):
        super(Diffusion, self).__init__()
        self.device = device
        self.timesteps = timesteps
        
        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        else:
            raise ValueError(f'unkown beta schedule {beta_schedule}')
        
        self.betas = beta_schedule_fn(self.timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0).to(self.device)

    def sample_timesteps(self, batch_size: int):
        return torch.randint(low=1, high=self.timesteps, size=(batch_size, ))
    
    def q_sample(self, x: torch.Tensor, t: torch.Tensor):
        alpha_bars_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        epsilon = torch.rand_like(x, device=self.device)
        
        noisy = torch.sqrt(alpha_bars_t) * x + torch.sqrt(1 - alpha_bars_t) * epsilon
        return noisy, epsilon