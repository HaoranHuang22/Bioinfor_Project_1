from torch.utils.data import Dataset, DataLoader
from utils.data import *
import numpy as np

def scale_down(coords, times=15):
    return coords / times

class CAcoordsDataset(Dataset):
    def __init__(self, pdb_chains, data_file_ca, data_file_res):
        pdb_chains = [str(x[0]) + str(x[1]) for x in pdb_chains]
        self.id = pdb_chains
        self.data = np.load(data_file_ca)
        self.res = np.load(data_file_res)

    def __len__(self):
        # Return the number of data points
        return len(self.data)
    
    def __getitem__(self, index):
        pdb_id = self.id[index] # pdb_id
        ca_coords = self.data[index]
        scaled_ca_coords = scale_down(ca_coords, times=15) # C alpha coordinates

        residues = self.res[index] # residues

        return pdb_id, scaled_ca_coords, residues 
    
class CAcoordsDataLoader(DataLoader):
    def __init__(self, pdb_chains, data_file_ca, data_file_res, batch_size=128, shuffle=True):
        dataset = CAcoordsDataset(pdb_chains, data_file_ca, data_file_res)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
    



    

