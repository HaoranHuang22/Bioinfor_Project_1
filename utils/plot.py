import nglview as nv
from ase import Atoms
from utils.data import *
import matplotlib.pyplot as plt
import torch

def visualize_pdb(pdb_chain, data_dir="../data/pdb"):
    """
    Funtion to visualize PDB file 

    Args:
        pdb_chain(tuple): pdb id and chain character
        data_dir(str): path to pdb directory

    Returns:
        A plot of C alpha atoms for given protein
    """
    #get pdb chain data:
    chain = get_pdb_chains(pdb_chain, data_dir)[1]

    view = nv.show_biopython(chain)

    return view

def visualize_ase(ca_coords):
    """
    Funtion to visualize C alpha atoms

    Args:
        ca_coords(np.array): C alpha coordinate
    Returns:
        A plot of C alpha atoms for given protein
    """
    ca = ['C'] * len(ca_coords)
    ca_atoms = Atoms(ca, ca_coords)

    view = nv.show_ase(ca_atoms)

    return view

def plt_Calpha(ax, pdb_chain_id, ca_coords):
    """
    Funtion to visualize C alpha atoms

    Args:
        ax(matplotlib.axes.Axes): axes object to use for plotting
        pdb_chain_id(tuple): (pdb id, chain character)
        ca_coords(np.array): C alpha coordinate
    Returns:
        A plot of C alpha atoms for given protein
    """
    if type(ca_coords) is torch.Tensor:
        ca_coords = ca_coords.detach().cpu().numpy()

    x_coords = ca_coords[:,0]
    y_coords = ca_coords[:,1]
    z_coords = ca_coords[:,2]
    pdb = pdb_chain_id[0]
    chain = pdb_chain_id[1]

    ax.plot(x_coords, y_coords, z_coords, 'r.', label='C alpha atoms')
    ax.plot(x_coords, y_coords, z_coords, 'b-', label='backbone')
    ax.set_title(pdb + " " + chain)
    #ax.legend()

def show_Calpha_plots(pdb_chains, ca_coords):
    """
    Make C alpha coordinates plot for given pdb chain id

    Args:
        pdb_chains(list): list of pdb id and chain character
        ca_coords(np.array): C alpha coordinates
    """
    if type(ca_coords) is torch.Tensor:
        ca_coords = ca_coords.detach().cpu().numpy()

    # Determine the number of rows and columns for the subplot grid
    n_chains = len(pdb_chains)
    n_rows = int(np.ceil(np.sqrt(n_chains)))
    n_cols = int(np.ceil(n_chains / n_rows))

    # Create a grid of subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10,10), subplot_kw=dict(projection='3d'))

    # Flatten the subplot grid so that we can iterate over it using a single loop
    axs = axs.flatten()

    # Iterate over the subplots and plot the C alpha atoms for each protein
    for i, ax in enumerate(axs):
        if i < n_chains:
            pdb_chain = pdb_chains[i]
            data = ca_coords[i]
            plt_Calpha(ax, pdb_chain, data)
        else:
            ax.axis('off')

    # Remove any unused subplots and adjust the layout
    #fig.delaxes(axs[len(pdb_chains):])
    plt.tight_layout()
    plt.show()

def show_diffusion_image(ax, ca_coords, t):

    if type(ca_coords) is torch.Tensor:
        ca_coords = ca_coords.detach().cpu().numpy()
    
    if type(t) is torch.Tensor:
        t = t.detach().cpu().numpy()

    x_coords = ca_coords[:,0]
    y_coords = ca_coords[:,1]
    z_coords = ca_coords[:,2]
    ax.plot(x_coords, y_coords, z_coords, 'r.', label='C alpha atoms')
    ax.plot(x_coords, y_coords, z_coords, 'b-', label='backbone')
    ax.set_title("Time:" + str(t))

def show_foward_diffusion(ddpm, pdb_chain, ca_coords, T):
    """
    Show the foward diffusion process
    """
    pdb = pdb_chain[:4]
    chain = pdb_chain[4]
    print("PDB id:" + pdb + " " + chain)
    n_images = 11
    n_rows = int(np.ceil(np.sqrt(n_images)))
    n_cols = int(np.ceil(n_images / n_rows))

    # Create a grid of subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8,8), subplot_kw=dict(projection='3d'))
    # Flatten the subplot grid so that we can iterate over it using a single loop
    axs = axs.flatten()

    # Iterate over the subplots and plot the C alpha atoms for each protein
    for i, ax in enumerate(axs):
        if i < n_images:
            t = torch.tensor(int(T / (n_images-1) * i))
            if t == 1000:
                t = 999
            noisy = ddpm.coord_q_sample(ca_coords, t)
            noisy = noisy.squeeze().squeeze() # dim: 20, 3
            show_diffusion_image(ax, noisy, t)
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_3d_scatter(ax, ca_coords, title):
    if type(ca_coords) is torch.Tensor:
        ca_coords = ca_coords.detach().cpu().numpy()
    
    x_coords = ca_coords[:,0]
    y_coords = ca_coords[:,1]
    z_coords = ca_coords[:,2]

    ax.plot(x_coords, y_coords, z_coords, 'r.', label='C alpha atoms')
    ax.plot(x_coords, y_coords, z_coords, 'b-', label='backbone')
    ax.set_title(title)













