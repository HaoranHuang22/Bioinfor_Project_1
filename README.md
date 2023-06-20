# Bioinfor_Project_1
- Replication of the structure and sequence diffusion model of [Protein Structure and Sequence Generation with Equivariant Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2205.15019) in Pytorch

## Data
53414 training pdb chains and 4372 test pdb chains

## Requrie Packages
```Shell
$ pip install invariant-point-attention
$ pip install roma
```
## Model Architecture(demo)
### Embedding Model(without constraints)
- Using ESM-1b pretrained model to get embedding vector for 20 types of amino acids, ESM-1b model returns a 1280 dim vector for each amino acid.
```python
from diffusion_model.embedd import get_single_representation

num_res = 20
embedding_dim = 1280
pdb_chain = ("12asA", "12e8H", ...) # (batch, )
res_label = torch.Tensor([[[1],[2], ...]]) # (batch, num_res, 1)
single_repr = get_single_representation(pdb_chain, res_label) # (batch, num_res, embedding_dim)
```

- Pair representation: C alpha distance matrix

```python
import torch

batch = 128
num_res = 20
ca_coords = torch.randn(batch, num_res, 3) # (batch, num_res, 3)
pair_repr = torch.cdist(ca_coords, ca_coords, p=2) # (batch, num_res, num_res)
pair_repr = pair_repr.unsqueeze(-1) # (batch, num_res, num_res, 1)
```
### Structure Diffusion Model
- Cosine schedule for beta

- Foward diffusion process for C alpha coordinates

- Foward diffusion process for quaternions

- IPABlock: Prediction model for rotaion matrix and translations from Alphafold2

```python
import torch
import roma
from diffusion_model.structure_diffusion_model import *

batch = 128
num_res = 20

diffusion = ProteinDiffusion(timesteps=1000, beta_schedule = 'cosine')
model = StructureModel(
                    input_single_repr_dim = 1280, 
                    input_pair_repr_dim = 1, 
                    dim = 128, 
                    structure_module_depth = 12, 
                    structure_module_heads = 4, 
                    point_key_dim = 4, 
                    point_value_dim = 4)

single_repr = torch.randn(batch, num_res, 1280) # (batch, num_res, embedding_dim)
pair_repr = torch.randn(batch, num_res, num_res, 1) # (batch, num_res, num_res, 1)

ca_coords = torch.randn(batch, num_res, 3) # (batch, num_res, 3)
q_0 = roma.rotmat_to_unitquat(R) # (batch, num_res, 4)

#foward diffusion
t = diffusion.sample_timesteps(batch_size = batch_size) #(batch, )
x_t = diffusion.coord_q_sample(ca_coords, t) # (batch, num_res, 3)
q_t = diffusion.quaternion_q_sample(q_0, t) # (batch, num_res, 4)

#model
pred_coords = model(single_repr, pair_repr, q_t, x_t)
```
### Sequence Diffusion Model



### Parameters
The following parameters were used in the structure diffusion model:
- Learning rate: {0.001}
- Batch size: {128}
- Epochs: {50}
- ESM embedding dim: {1280}
- Time steps: {1000}
