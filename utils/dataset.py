from torch.utils.data import Dataset, DataLoader
from utils.data import *
import numpy as np
import pandas as pd
import ast

def scale_down(coords, times=15):
    return coords / times

class BackboneCoordsDataset(Dataset):
    def __init__(self, pdb_chains, data_file_coords, data_file_res, seq_length = int):
        pdb_chains = [str(x[0]) + str(x[1]) for x in pdb_chains]
        self.id = pdb_chains
        self.data = np.load(data_file_coords)
        self.res = np.load(data_file_res)

    def __len__(self):
        # Return the number of data points
        return len(self.data)
    
    def __getitem__(self, index):
        pdb_id = self.id[index]
        atom_coords = self.data[index]
        scale_down_coords = scale_down(atom_coords, times=15) # backbone atom coordinates
        residues = self.res[index] # residue

        return pdb_id, residues, scale_down_coords
    
class BackboneCoordsDataLoader(DataLoader):
    def __init__(self, pdb_chains, data_file_coords, data_file_res, seq_length, batch_size=128, shuffle=True):
        dataset = BackboneCoordsDataset(pdb_chains, data_file_coords, data_file_res, seq_length)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

def subsample_protein_data(pd_data, fixed_length=16):
    """
    Args:
        pd_data(pd.DataFrame): protein data
        fixed_length(int): fixed length of protein sequence

    Return:
        pd_data(list): protein data with fixed length
    """
    subsample_data = []

    for index, row in pd_data.iterrows():
        pdb_id = row['pdb_id']
        res_label = np.array(eval(row['res_label'])).reshape(-1) # reshape to 1D array
        atom_coords = np.array(eval(row['atom_coords']))

        if len(res_label) > len(atom_coords):
            gap = len(res_label) - len(atom_coords)
            res_label = res_label[gap:]
            
        if len(res_label) > fixed_length: # if the length of protein sequence is longer than fixed length, then subsample
            padded_res_label = res_label[:fixed_length]
            padded_atom_coords = atom_coords[:fixed_length]

        subsample_data.append([pdb_id, padded_res_label, padded_atom_coords])

    return subsample_data


class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pdb_id, res_label, atom_coords = self.data[index]
        atom_coords = scale_down(atom_coords, times=15) # backbone atom coordinates
        # mask out the padding value
        mask = np.where(res_label == 0, 0, 1)

        return pdb_id, res_label, atom_coords, mask


    

