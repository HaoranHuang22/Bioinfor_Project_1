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

    def sample_timesteps(self, batch_size: int, device):
        t = torch.randint(low=1, high=self.timesteps, size=(batch_size, ), device=device)

        pt = torch.ones_like(t).float() / self.timesteps
        return t, pt
    
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
        mask_prob = mask_prob.repeat(self.res_num, 1).T # expand to (batch_size, num_res)
        
        mask = torch.bernoulli(mask_prob).bool() # create mask matrix with probability mask_prob
        x_t = x_0.masked_fill(mask, 21) # mask x with mask matrix, and fill masked position with 21

        x_0_ignore[torch.bitwise_not(mask)] = -1 # ignore the position that is not masked

        return x_t, x_0_ignore, mask
    
    def p_sample(self, model: nn.Module, batch_size: int, device, pair_repr: torch.Tensor, rotations: torch.Tensor, translations: torch.Tensor):
        """
        Args:
            model(nn.Module): model for sampling
            batch_size(int): batch size
            device(torch.device): device
            pair_repr(torch.Tensor): dim -> (batch_size, num_res, num_res)
            rotations(torch.Tensor): dim -> (batch_size, num_res, 3, 3)
            translations(torch.Tensor): dim -> (batch_size, num_res, 3)
            
        Returns:
            x_t(torch.Tensor): dim -> (batch_size, num_res)
        """
        x_t = torch.ones(batch_size, self.res_num, device=device, dtype=torch.int64) * 21

        unmasked = torch.zeros_like(x_t, device=device).bool()
        sample_steps = list(range(1, self.timesteps + 1))

        for t in reversed(sample_steps):
            t = torch.full((batch_size, ), t, device=device, dtype=torch.int64)
            # where to unmask
            t_mask = t.reshape(batch_size, 1)
            t_mask = t_mask.expand(batch_size, self.res_num)

            changes = torch.rand_like(x_t.float()) < 1.0 / t_mask.float()
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))

            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)
            x_0_logits = model(x_t.float(), pair_repr, rotations, translations)
            index = torch.argmax(x_0_logits, dim=2)

            x_t.long()[changes] = index.long()[changes]

        return x_t



class SequenceModel(nn.Module):
<<<<<<< HEAD
    def __init__(self, input_single_repr_dim = 1, input_pairwise_repr_dim = 1, dim = 64, structure_module_depth = 12, 
=======
    def __init__(self, input_single_repr_dim = 1, dim = 64, structure_module_depth = 12, 
>>>>>>> 14fe12319cf4514e28edc6f2b794d6d0379ee263
                 structure_module_heads = 4, point_key_dim = 4, point_value_dim = 4):
        super().__init__()
        
        self.single_repr = nn.Linear(input_single_repr_dim, dim)
<<<<<<< HEAD
        self.pair_repr = nn.Linear(input_pairwise_repr_dim, dim)
=======
>>>>>>> 14fe12319cf4514e28edc6f2b794d6d0379ee263

        self.structure_module_depth = structure_module_depth
        self.ipa_block =IPABlock(dim=dim, heads = structure_module_heads, 
                                    point_key_dim = point_key_dim, point_value_dim = point_value_dim, 
                                    require_pairwise_repr = False)
        
        self.to_points = nn.Linear(dim, 20)
    
    def forward(self, single_repr:torch.Tensor, pair_repr:torch.Tensor, rotations:torch.Tensor, translations:torch.Tensor):
        """
        Args:
            single_repr(torch.Tensor): dim -> (batch_size, num_res)
            rotations(torch.Tensor): dim -> (batch_size, num_res, 3, 3)
            translations(torch.Tensor): dim -> (batch_size, num_res, 3)
        Return:
            single_repr(torch.Tensor): dim -> (batch_size, num_res)
        """
        single_repr = single_repr.unsqueeze(-1) # dim -> (batch_size, num_res, 1)
        pair_repr = pair_repr.unsqueeze(-1) # dim -> (batch_size, num_res, num_res, 1)
        
        single_repr = self.single_repr(single_repr) # dim -> (batch_size, num_res, dim)
        pair_repr = self.pair_repr(pair_repr) # dim -> (batch_size, num_res, num_res, dim)

        for i in range(self.structure_module_depth):
            is_last = i == (self.structure_module_depth - 1)

            if not is_last:
                rotations = rotations.detach()
            
            single_repr = self.ipa_block(single_repr, pairwise_repr = pair_repr, rotations = rotations, translations = translations)

        single_repr = self.to_points(single_repr) # dim -> (batch_size, num_res, 20)
        
        return single_repr
            


