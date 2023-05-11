import torch

def sequence_recovery_rate(pred_seq, target_seq, mask):
    """
    Args:
        pred_seq(torch.Tensor): dim -> (batch_size, num_res)
        target_seq(torch.Tensor): dim -> (batch_size, num_res)
        mask(torch.Tensor): dim -> (batch_size, num_res)
    Return:
        recovery_rate(torch.Tensor): dim -> (batch_size, )
    """
    
    mask = mask.bool()

    correct_non_padding = (pred_seq == target_seq) & mask

    num_correct = torch.sum(correct_non_padding, dim=1)

    num_non_padding_position = torch.sum(mask, dim=1)

    recovery_rate = num_correct.float() / num_non_padding_position.float()
    

    return recovery_rate