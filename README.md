# Bioinfor_Project_1
- Replication of the structure diffusion model of [Protein Structure and Sequence Generation with Equivariant Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2205.15019) in Pytorch

## Input data
Subsample 20 amino acid for each protein
- pdb id and chain character 
- residue type
- N, C alpha, C atom coordinates


## Model Architecture
### Embedding Model(without constraints)
- Using ESM-1b pretrained model to get embedding vector for 20 types of amino acids, ESM-1b model returns a 1280 dim vector for each amino acid.
```python
from diffusion_model.embedd import get_single_representation

num_res = 20
embedding_dim = 1280
pdb_chain = ("12asA", "12e8H", ...) # (batch, )
res_label = torch.Tensor() # (batch, num_res, 1)
single_repr = get_single_representation(pdb_chain, res_label) # (batch, num_res, embedding_dim)
```
- pair representation: C alpha distance matrix
```python
import torch

batch = 128
num_res = 20
ca_coords = torch.randn(batch, num_res, 3) # (batch, num_res, 3)
pair_repr = torch.cdist(ca_coords, ca_coords, p=2) # (batch, num_res, num_res)
pair_repr = pair_repr.unsqueeze(-1) # (batch, num_res, num_res, 1)
```
### Diffusion Model
- Cosine schedule for beta

- Foward diffusion process for C alpha coordinates

- Foward diffusion process for quaternions

- IPABlock: Prediction model for rotaion matrix and translations from Alphafold2

### Parameters
The following parameters were used in the structure diffusion model:
- Learning rate: {0.001}
- Batch size: {128}
- Epochs: {50}
- ESM embedding dim: {1280}
- Time steps: {1000}
