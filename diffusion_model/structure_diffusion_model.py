import torch 
import torch.nn as nn
import torch.nn.functional as F

import common.res_infor

import esm


def Embedder(pdb_chain, res_label, ca_coords):
    """
    Function to get single representation and pair representation

    Args:
        res_label(torch.Tensor): dim -> (batch_size, num_res, 1)
        ca_coords(torch.Tensor): dim -> (batch_size, num_res, 3)

    Returns:
        single_repr(torch.Tensor): dim -> (batch_size, embedding_dim)
        pairwise_repr(torch.Tensor): dim -> (batch_size, num_res, num_res)
    """
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
