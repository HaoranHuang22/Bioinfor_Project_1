import roma
import torch
from einops import rearrange

def computeFAPE(quaternions_pred, translation_pred, x_pred, quaternions_gt, translation_gt, x_gt, Z = 10.0, d_clamp=10.0, epsilon=10e-4):
    """
    modified from https://github.com/wangleiofficial/FAPEloss/blob/main/fape.py
    Args:
        quaternions_pred(torch.Tensor): dim -> (batch_size, num_res, 4)
        translation_pred(torch.Tensor): dim -> (batch_size, num_res, 3)
        x_pred(torch.Tensor): dim -> (batch_size, num_res, 3)
        quaternions_gt(torch.Tensor): dim -> (batch_size, num_res, 4)
        translation_gt(torch.Tensor): dim -> (batch_size, num_res, 3)
        x_gt(torch.Tensor): dim -> (batch_size, num_res, 3)
    """
    rotations_pred = roma.unitquat_to_rotmat(quaternions_pred)
    rotations_gt = roma.unitquat_to_rotmat(quaternions_gt)

    delta_x_pred = rearrange(x_pred, 'b j t -> b j () t') - rearrange(translation_pred, 'b i t -> b () i t')
    delta_x_gt = rearrange(x_gt, 'b j t -> b j () t') - rearrange(translation_gt, 'b i t -> b () i t')


    X_pred = torch.einsum('b i k q, b j i t -> b i j q', rotations_pred, delta_x_pred)
    X_gt = torch.einsum('b i k q, b j i t -> b i j q', rotations_gt, delta_x_gt)

    distance = torch.sqrt(torch.sum((X_pred - X_gt)**2, dim = -1) + epsilon)
    distance = torch.clamp(distance, max = d_clamp) / Z

    FAPE_loss = torch.mean(distance)

    return FAPE_loss


    
    
