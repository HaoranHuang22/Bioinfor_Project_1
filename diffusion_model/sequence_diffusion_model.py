import torch 
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F

import math

import roma
from invariant_point_attention import IPABlock

def rigidFrom3Points(x1, x2, x3):
    """
    Construct of frames from groud truth atom positions
    From alphaFold2 supplementary Algorithm {21} Rigid from 3 points using the Gram–Schmidt process
    Args:
        x1(torch.Tensor): N atom coordinates. dim -> (batch_size, num_res, 3)
        x2(torch.Tensor): C alpha atom coordinates. -> (batch_size, num_res, 3)
        x3(torch.Tensor): C atom coordinates. -> (batch_size, num_res, 3)
    
    Return:
        Rotations(tensor.Tensor): Rotation matrix. dim -> (batch_size, num_res, 3, 3)
        translations(tensor.Tensor): translation vector. dim -> (batch_size, num_res, 3)
    """
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / torch.linalg.norm(v1, dim=2, keepdim=True)
    u2 = v2 - torch.einsum('bij,bij->bi', v2, e1).unsqueeze(-1) * e1 / torch.einsum('bij,bij->bi', e1, e1).unsqueeze(-1)
    e2 = u2 / torch.linalg.norm(u2, dim=2, keepdim=True)
    e3 = torch.cross(e1, e2)
    R = torch.stack([e1, e2, e3], dim=2) # column vector matrix
    t = x2

    return R, t

def one_hot_encoding(x, vocab_size = 21):
    """
    One hot encoding for amino acid sequence

    Args:
        x(torch.Tensor): amino acid sequence. dim -> (batch_size, num_res, 1)
        vocab_size(int): vocabulary size

    Return:
        x_one_hot(torch.Tensor): one hot encoded amino acid sequence. dim -> (batch_size, num_res, vocab_size)
    """
    one_hot_batch = torch.zeros(x.size(0), x.size(1), vocab_size)
    x_one_hot = one_hot_batch.scatter_(2, x, 1)
    return x_one_hot

class SequenceDiffusion(nn.Module):
    def __init__(self, timesteps=100, res_num = 20, device = None):
        super(SequenceDiffusion,self).__init__()
        self.device = device
        self.res_num = res_num
        self.timesteps = timesteps

    def sample_timesteps(self, batch_size: int):
        return torch.randint(low=1, high=self.timesteps + 1, size=(batch_size, ))
    
    def seq_q_sample(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        Foward diffusion process for sequence data

        Args:
            x(torch.Tensor): dim -> (batch_size, num_res)
            t(torch.Tensor): dim -> (batch_size, )

        Return:
            masked_x(torch.Tensor): dim-> (batch_size, num_res)
        """
        x_0_ignore = x_0.clone()

        mask_prob = t / self.timesteps
        mask_prob = mask_prob.repeat(self.res_num, 1).T # reshape to (batch_size, num_res)

        mask = torch.bernoulli(mask_prob).bool() # create mask matrix with probability mask_prob
        x_t = x_0.masked_fill(mask, 21) # mask x with mask matrix, and fill masked position with 21
        x_0_ignore[torch.bitwise_not(mask)] = -1 # ignore the position that is not masked
        
        return x_t, x_0_ignore
    


