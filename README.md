# Bioinfor_Project_1

## Input data
The dataset consists of three parts (pdb id with chain character, residue type, N, C alpha, C atom coordinates), subsample 20 aa for each protein.

## Model Architecture
### Embedding Model
- single representation: ESM-1b embedding for each amino acid
- pair representation: C alpha distance matrix

### Diffusion Model
- Cosine schedule for beta

- Foward diffusion process for C alpha coordinates

- Foward diffusion process for quaternions

- IPABlock: Prediction model for rotaion matrix and translations from Alphafold2

### Parameters
The following parameters were used in the coordinate diffusion model:
- Learning rate: {0.001}
- Batch size: {128}
- Epochs: {50}
- ESM embedding dim: {1280}
- Time steps: {1000}
