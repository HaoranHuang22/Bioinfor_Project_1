# Bioinfor_Project_1

## Dataload
The dataset consists of three parts (pdb id with chain character, centered C alpha coordinates that are scaled down 15X, residue type).
Subsample 20 aa for each protein

## Model Architecture
### Diffusion Model
- Cosine schedule for beta

- Positional Encoding

- Normal distribution noise

- UNet: The coordinate diffusion model used a castrated version U-Net neural network predicts noise error.

### Parameters
The following parameters were used in the coordinate diffusion model:
- Learning rate: {0.001}
- Batch size: {128}
- Epochs: {100}
- Time embedding dimension: {128}
- Time steps: {100}
