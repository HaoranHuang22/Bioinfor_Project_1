import torch 
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F

import math

import roma
from invariant_point_attention import IPABlock
from loss.FAPE import computeFAPE

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

class ProteinDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_schedule = 'cosine', device = None):
        super(ProteinDiffusion, self).__init__()
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
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0)

    def sample_timesteps(self, batch_size: int):
        return torch.randint(low=1, high=self.timesteps, size=(batch_size, ))
    
    def coord_q_sample(self, x: torch.Tensor, t: torch.Tensor):
        """
        Foward diffusion process for C alpha coordinates

        Args:
            x(torch.Tensor): dim -> (batch_size, num_res, 3)
            t(torch.Tensor): dim -> (batch_size, )

        Return:
            noisy(torch.Tensor): dim-> (batch_size, num_res, 3)
        """
        alpha_bars_t = self.alpha_bars[t].view(-1, 1, 1) # dim -> (batch_size, 1, 1)
        epsilon = torch.rand_like(x)
        
        noisy = torch.sqrt(alpha_bars_t) * x + torch.sqrt(1 - alpha_bars_t) * epsilon
        return noisy
    
    def quaternion_q_sample(self, q_0:torch.Tensor, t: torch.Tensor):
        """
        Foward diffusion process for unit quaternions

        Args:
            q_0(torch.Tensor): dim -> (batch_size, num_res, 4)
            t(torch.Tensor): dim -> (batch_size, )

        Return:
            q_t(torch.Tensor): dim -> (batch_size, num_res, 4)
        """
        batch_size, num_res = q_0.shape[0], q_0.shape[1]

        alpha_t = self.alphas[t]
        q_T = roma.random_unitquat(size=(batch_size, num_res)) # dim -> (batch_size, num_res, 4)
        q_interpolated = roma.utils.unitquat_slerp(q_0.double(), q_T.double(), alpha_t) #(t_size, batch_size, 20, 4)
        # create empty result tensor
        q_t = torch.empty_like(q_T)
        # loop over t_size and extract desired batch
        for t in range(len(alpha_t)):
            batch = q_interpolated[t, t % batch_size]
            q_t[t] = batch # dim -> (t_size, num_res, 4)
        
        return q_t
    
    def sample_p(self, model:nn.Module, single_repr: torch.Tensor, pair_repr: torch.Tensor):
        """
        sample from predicted x_0 and q_0
        Args:
            single_repr(torch.Tensor): dim -> (batch_size, num_res, 1280)
            pair_repr(torch.Tensor): dim -> (batch_size, num_res, num_res)
        Return:
            x_list(list): list of x_t, dim -> (timesteps, batch_size, num_res, 3)
            q_list(list): list of q_t, dim -> (timesteps, batch_size, num_res, 4)
        """
        batch_size, num_res = single_repr.shape[0], single_repr.shape[1]

        x_t = torch.randn(size=(batch_size, num_res, 3), device = self.device).to(torch.float32)
        q_t = roma.random_unitquat(size=(batch_size, num_res), device = self.device).to(torch.float32)
        
        sample_steps = list(range(1, self.timesteps + 1))
        
        x_hat, q_hat = model(single_repr, pair_repr, q_t, x_t) 

        return x_hat, q_hat
            

class StructureModel(nn.Module):
    def __init__(self, input_single_repr_dim = 1280, input_pair_repr_dim = 1, dim = 128, 
                 structure_module_depth = 12, structure_module_heads = 4, point_key_dim = 4, point_value_dim = 4):
        super().__init__()

        self.single_repr = nn.Linear(input_single_repr_dim, dim)

        
        self.pair_repr = nn.Linear(input_pair_repr_dim, dim)

        self.structure_module_depth = structure_module_depth
        self.ipa_block =IPABlock(dim=dim, heads = structure_module_heads, 
                                 point_key_dim = point_key_dim, point_value_dim = point_value_dim)
        
        self.to_points = nn.Linear(dim, 6)
        self.to_quaternion_update = nn.Linear(dim, 6)
        

    def forward(self, single_repr:torch.Tensor, pair_repr:torch.Tensor, quaternions:torch.Tensor, translations:torch.Tensor):
        """
        Args:
            single_repr: dim -> (batch_size, num_res, embedding_dim)
            pair_repr: dim -> (batch_size, num_res, num_res)
            quaternions: dim -> (batch_size, num_res, 4)
            translation: dim -> (batch_size, num_res, 3)
        
        Returns:
            rotations: dim -> (batch_size, num_res, 3, 3)
            translations: dim -> (batch_size, num_res, 3)
            points_local: dim -> (batch_size, num_res, 3)
        """

        pair_repr = pair_repr.unsqueeze(-1) # dim -> (batch_size, num_res, num_res, 1)
        quaternions_gt = quaternions.clone()
        translations_gt = translations.clone()

        # Linear Foward
        single_repr = self.single_repr(single_repr)
        pair_repr = self.pair_repr(pair_repr)

        auxiliary_loss = 0

        for i in range(self.structure_module_depth):
            is_last = i == (self.structure_module_depth - 1)

            rotations = roma.unitquat_to_rotmat(quaternions) 

            if not is_last:
                rotations = rotations.detach()
    
            single_repr = self.ipa_block(single_repr, pairwise_repr = pair_repr, 
                                         rotations = rotations, translations = translations)
            
            # update quaternion and translation
            quaternion_update, translation_update = self.to_quaternion_update(single_repr).chunk(2, dim = -1) # (batch_size, num_res, 6) -> (batch_size, num_res, 3), (batch_size, num_res, 3)
            quaternion_update = F.pad(quaternion_update, (0, 1), value = 1.) # dim -> (batch_size, num_res, 4), roma expects quatations to be in (x, y, z, w) format
            quaternion_update = quaternion_update / torch.linalg.norm(quaternion_update, dim = -1, keepdim = True) # normalize quaternion to unit quaternion

            quaternions = roma.quat_product(quaternions, quaternion_update)
            translations = translations + einsum('b n c, b n c r -> b n r', translation_update, rotations)

            # auxiliary loss
            auxiliary_loss += computeFAPE(quaternions, translations, translations, quaternions_gt, translations_gt, translations_gt)

        # convert to points
        n_points_local, c_points_local = self.to_points(single_repr).chunk(2, dim = -1) # dim -> (batch_size, num_res, 3), (batch_size, num_res, 3)
        rotations = roma.unitquat_to_rotmat(quaternions) # dim -> (batch_size, num_res, 3, 3)
        auxiliary_loss = auxiliary_loss / self.structure_module_depth

        return quaternions, translations, n_points_local, c_points_local, auxiliary_loss

