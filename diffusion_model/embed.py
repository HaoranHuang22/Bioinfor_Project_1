import esm
import torch
from common.res_infor import label_res_dict


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