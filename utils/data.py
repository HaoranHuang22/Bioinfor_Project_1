from Bio.PDB import *
import os
import numpy as np
import common.res_infor
from Bio.PDB.DSSP import dssp_dict_from_pdb_file # only for mac/linux 
import csv

def read_domain_ids_per_chain_from_txt(txt_file):
    """
    Function to get all pdb id and corresponding chain id

    Args:
        txt_file: file name
    
    Returns:
        list of pdb_id
        list of (pdb_id, chain_character)
    """
    
    pdbs = []
    pdb_chains = []
    with open(txt_file, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip('\n').split()
            pdbs.append(line[0][:4]) #CHARACTERS 1-4: PDB Code
            if line[0][4] != '0':
                pdb_chains.append((line[0][:4],line[0][4])) #CHARACTER 5: Chain Character
            else:
                pdb_chains.append((line[0][:4],0))
    return pdbs, pdb_chains

def download_pdb(pdb, data_dir="./pdb"):
    """
    Function to download PDB file from PDB

    Args:
        pdb: pdb id
    """
    pdbl = PDBList()
    f = data_dir + "/" + "pdb" + pdb + ".ent"
    pdbl.retrieve_pdb_file(pdb, pdir=data_dir, file_format='pdb')
    if not os.path.isfile(f):
        f = data_dir + "/" + pdb + ".pdb"
        if not os.path.isfile(f):
            os.system("wget -O {} https://files.rcsb.org/download/{}.pdb".format(f, pdb.upper()))
    return f

def get_pdb_chains(pdb_chains,  data_dir="../data/pdb"):
    """
    Function to load pdb structure via Biopython and extract all chains

    Args:
        pdb_chains(tuple): pdb id and chain character
        data_dir(str): path to pdb directory

    Returns:
        (pdb id, chain)
    """
    pdb = pdb_chains[0]
    chain = pdb_chains[1]
    if pdb == "5exc":
        chain = 'ee'
    f = data_dir + "./" + "pdb" + pdb + ".ent"
    parser = PDBParser(QUIET=True)
    if os.path.isfile(f):
        structure = parser.get_structure(pdb, f)
    elif os.path.isfile(data_dir + "/" + pdb + ".pdb"):
        f = data_dir + "/" + pdb + ".pdb"
        structure = parser.get_structure(pdb, f)
    else:
        f = data_dir + "/" + pdb + ".cif"
        structure = MMCIFParser(QUIET=True).get_structure(pdb, f) # for large protein that can't be formated in pdb
    model = structure[0]
    if chain == 0: # if a model doesn't have any chain, then output the model directly
        return((pdb,model))
    else:
        return((pdb,model[chain]))

def get_pdb_data(pdb_chains, data_dir="../data/pdb"):
    """
    Function to get C alpha coordinate from pdb structures

    Args:
        pdb_chains(tuple): pdb id and chain character
        data_dir(str): path to pdb directory

    Returns:
        CA_coords (np.array): num_Calpha atoms x 3 coordinates of all retained C alpha atoms in structure
        res_label (np.array): num_res x 1 residue type labels (amino acid type) for all residues
    """
    #get pdb chain data
    chain = get_pdb_chains(pdb_chains)[1]

    CA_coords = []
    res_label = []

    for res in chain.get_residues():
        res_name = res.get_resname()
        
        # if residue is an amino acid, add its label, get the C alpha coordinates 
        if res_name in common.res_infor.res_label_dict.keys():
            # get residue type
            res_type = common.res_infor.res_label_dict[res_name]
            res_label.append(res_type)
            try:
            # get C alpha atomic coordinate
                ca = res['CA'].get_coord().tolist()
                CA_coords.append(ca)
            except KeyError:
                pass
    
    CA_coords = np.array(CA_coords)
    res_label = np.array(res_label).reshape(len(res_label),1)
    

    return CA_coords, res_label

def get_Backbone_atom_coords(pdb_chains, data_dir="../data/pdb"):
    """
    Function to get N, C alpha, C coordinate for each residue

    Args:
        pdb_chains(tuple): pdb id and chain character
        data_dir(str): path to pdb directory

    Returns:
        atom_coords (np.array): num_res atoms x 3 atoms x 3 coordinates of all retained C alpha atoms in structure
        res_label (np.array): num_res x 1 residue type labels (amino acid type) for all residues
    """
    #get pdb chain data
    chain = get_pdb_chains(pdb_chains)[1]
    
    atom_coords = []
    res_label = []

    for res in chain.get_residues():
        res_name = res.get_resname()

        # if residue is an amino acid, add its label, get the N, C alpha, C coordinates
        if res_name in common.res_infor.res_label_dict.keys():
            # get residue type
            res_type = common.res_infor.res_label_dict[res_name]
            res_label.append(res_type)
            try:
                n = res['N'].get_coord().tolist()
                ca = res['CA'].get_coord().tolist()
                c = res['C'].get_coord().tolist()
                atom_coords.append([n,ca,c])
            except KeyError:
                pass

    atom_coords = np.array(atom_coords)
    res_label = np.array(res_label).reshape(len(res_label), 1)

    return atom_coords, res_label

def load_backbone_coords(pdb_chains):
    """
    Function to get 

    Args:
        pdb_chains(list): a list of pdb id and chain character

    Returns:
        data(np.array): num_sample x num_res X 3 atoms x 3 coordinates of all retained C alpha atoms in structure
    """
    data_coords = []
    data_res = []
    n = len(pdb_chains)
    count = 0
    for pdb_chain in pdb_chains:
        atom_coords, res_label = get_Backbone_atom_coords(pdb_chain)
        data_coords.append(atom_coords)
        data_res.append(res_label)
        count += 1
        print(str(count) + "/" + str(n))

    data_coords = np.array(data_coords)
    data_res = np.array(data_res)
    return data_coords, data_res

def get_phi_psi_angles(pdb_chain, data_dir = "../data/pdb"):
    """
    Args:
        pdb_chains(tuple): pdb id and chain character
        data_dir(str): path to pdb directory
    Returns:
        phi_psi_angles(np.array): num_atoms x 2 angles of phi and psi
    """
    # get pdb chain data
    chain = get_pdb_chains(pdb_chain)[1]
    phi_psi_angles = []

    # Create a list of  polypeptide objects
    ppb = PPBuilder()
    pp_list = ppb.build_peptides(chain)

    # Get phi and psi angles
    for pp in pp_list:
        for phi, psi in pp.get_phi_psi_list():
            phi_psi_angles.append([phi, psi])
    
    phi_psi_angles = np.array(phi_psi_angles)
    return phi_psi_angles


def load_all_pdb_data_to_list(pdb_chains, data_dir="../data/pdb"):
    """
    Args:
        pdb_chains(list): a list of pdb id and chain character
        data_dir(str): path to pdb directory
    Returns:
        data(list): a list of tuples of pdb id and chain character, residue labels, atom coordinates, phi and psi angles
    """
    data = []
    n = len(pdb_chains)
    count = 0
    for pdb_chain in pdb_chains:
        atom_coords, res_label = get_Backbone_atom_coords(pdb_chain)
        phi_psi_angles = get_phi_psi_angles(pdb_chain)
        data.append((pdb_chain, res_label, atom_coords, phi_psi_angles))
        count += 1
        print(str(count) + "/" + str(n))
    return data

def save_data_to_csv(data, csv_file):
    """
    Args:
        data(list): a list of tuples of pdb id and chain character, residue labels, atom coordinates, phi and psi angles
        csv_file(str): path to csv file
    
    """
    with open(csv_file, mode='w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        # write header
        writer.writerow(["pdb_id", "res_label", "atom_coords", "phi_psi_angles"])
        # write data
        for pdb_id, res_label, atom_coords, phi_psi_angles in data:
            writer.writerow([pdb_id, res_label, atom_coords, phi_psi_angles])
            

def get_chi_angles(pdb_chains, data_dir="../data/pdb"):
    """
    Function to get four kinds of data from pdb structures

    Args:
        pdb_chains(tuple): pdb id and chain character
        data_dir(str): path to pdb directory

    Returns:
        chi_angles (np.array): num_Calpha atoms x 4 chi angles
    """
    #get pdb chain data
    chain = get_pdb_chains(pdb_chains)[1]

    chi_angles = []

    for res in chain.get_residues():
        res_name = res.get_resname()
        
        # if residue is an amino acid, add its label, get the C alpha coordinates and side-chains angle
        if res_name in common.res_infor.res_label_dict.keys():
            # get residue type
            res_type = common.res_infor.res_label_dict[res_name]

            # get C alpha atomic coordinate
            ca = res['CA'].get_coord().tolist()
    
            # get rotamer chi angles
            # amino acid ALA and GLY don't have chi angles
            if res_name == "ALA" or res_name == "GLY":
                chi = [0, 0, 0, 0]
            else:
                chi = []
                n = res['N'].get_vector()
                ca = res['CA'].get_vector()
                if "chi_1" in common.res_infor.chi_dict[res_name].keys():
                    cb = res['CB'].get_vector()
                    cg = res[common.res_infor.chi_dict[res_name]['chi_1']].get_vector()
                    chi_1 = calc_dihedral(n, ca, cb, cg)
                    chi.append(chi_1)
                    
                    if "chi_2" in common.res_infor.chi_dict[res_name].keys():
                        cd = res[common.res_infor.chi_dict[res_name]['chi_2']].get_vector()
                        chi_2 = calc_dihedral(ca, cb, cg, cd)
                        chi.append(chi_1)

                        if "chi_3" in common.res_infor.chi_dict[res_name].keys():
                            ce = res[common.res_infor.chi_dict[res_name]['chi_3']].get_vector()
                            chi_3 = calc_dihedral(cb, cg, cd, ce)
                            chi.append(chi_3)

                            if "chi_4" in common.res_infor.chi_dict[res_name].keys():
                                cz = res[common.res_infor.chi_dict[res_name]['chi_4']].get_vector()
                                chi_4 = calc_dihedral(cg, cd, ce, cz)
                                chi.append(chi_4)
                            else:
                                chi.append(0)
                        else:
                            chi.extend([0,0])
                    else:
                        chi.extend([0,0,0])
                else:
                    chi = [0,0,0,0]
            chi_angles.append(chi)

    chi_angles = np.array(chi_angles)

    return chi_angles

def load_CA_coords(pdb_chains):
    """
    Function to get 

    Args:
        pdb_chains(list): a list of pdb id and chain character

    Returns:
        data(np.array): num_sample x num_Calpha atoms x 3 coordinates of all retained C alpha atoms in structure
    """
    data_ca = []
    data_res = []
    n = len(pdb_chains)
    count = 0
    for pdb_chain in pdb_chains:
        CA_coords, res_label = get_pdb_data(pdb_chain)
        data_ca.append(CA_coords)
        data_res.append(res_label)
        count += 1
        print(str(count) + "/" + str(n))

    data_ca = np.array(data_ca)
    data_res = np.array(data_res)
    return data_ca, data_res


def get_secondary_structure(pdb_chains, data_dir = "./pdb"):
    """
    Function to get secondary structure topology of proteins, only on mac/linux

    Args:
        pdb_chains(tuple): pdb id and chain character
        data_dir(str): path to pdb directory

    Returns:
        secondary_structure_block(list): a list of 'H'(helix), 'E'(sheet), or 'L'(loop)
        amino_acid(list): a list of amino acid
    """
    pdb = pdb_chains[0]
    chain = pdb_chains[1]
    parser = PDBParser(QUIET=True)
    ss_list = []
    aa_list = []

    f = data_dir + "/" + "pdb" + pdb + ".ent"
    if os.path.isfile(f):
        structure = parser.get_structure(pdb, f)
    else:
        f = data_dir + "/" + pdb + ".pdb"
        structure = parser.get_structure(pdb, f)
    
    
    model = structure[0]
    dssp = DSSP(model, f)
    a_key = list(dssp.keys())
    for key in a_key:
        if key[0] == chain:
            aa = dssp[key][1]
            ss = dssp[key][2]
            # (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
            # NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
            # NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
            if ss == "E" or ss == "B": # strand (E, B)
                ss = "E"
            elif ss == "H" or ss == "G" or ss == "I": # helix (G, H, I)
                ss = "H"
            else:
                ss = "L"
            ss_list.append(ss)
            aa_list.append(aa)
    
    return aa_list, ss_list

