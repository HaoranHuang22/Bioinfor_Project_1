from torch.optim import Adam
from tqdm import tqdm

from utils.data import read_domain_ids_per_chain_from_txt
from utils.dataset import *
from diffusion_model.structure_diffusion_model import *



if __name__ == "__main__":
    train_pdbs, train_pdb_chains = read_domain_ids_per_chain_from_txt('./data/train_domains.txt')
    test_pdbs, test_pdb_chains = read_domain_ids_per_chain_from_txt('./data/test_domains.txt')
