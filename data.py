from Bio.PDB import *
import os
import numpy as np
import common.res_infor

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

def get_pdb_chains(pdb_chains,  data_dir="./pdb"):
    """
    Function to load pdb structure via Biopython and extract all chains

    Args:
        pdb_chains(tuple): pdb id and chain character
        data_dir(str): path to pdb directory

    returns:
        (pdb id, chain)
    """
    pdb = pdb_chains[0]
    chain = pdb_chains[1]
    f = data_dir + "./" + "pdb" + pdb + ".ent"
    parser = PDBParser(QUIET=True)
    if os.path.isfile(f):
        structure = parser.get_structure(pdb, f)
    else:
        f = data_dir + "/" + pdb + ".pdb"
        structure = parser.get_structure(pdb, f)
    model = structure[0]
    if chain == 0: # if a model doesn't have any chain, then output the model directly
        return((pdb,model))
    else:
        return((pdb,model[chain]))

def get_pdb_data(pdb_chains, data_dir="./pdb"):
    """
    Function to get four kinds of data from pdb structures

    Args:
        pdb_chains(tuple): pdb id and chain character
        data_dir(str): path to pdb directory

    Returns:
        CA_coords (np.array): num_Calpha atoms x 3 coordinates of all retained C alpha atoms in structure
        res_label (np.array): num_res x 1 residue type labels (amino acid type) for all residues
        chi_angles (np.array): num_Calpha atoms x 4 chi angles
    """
    #get pdb chain data
    chain = get_pdb_chains(pdb_chains)[1]

    CA_coords = []
    res_label = []
    chi_angles = []

    for res in chain.get_residues():
        res_name = res.get_resname()
        
        # if residue is an amino acid, add its label, get the C alpha coordinates and side-chains angle
        if res_name in common.res_infor.res_label_dict.keys():
            print(res_name)
            # get residue type
            res_type = common.res_infor.res_label_dict[res_name]
            res_label.append(res_type)

            # get C alpha atomic coordinate
            ca = res['CA'].get_coord().tolist()
            CA_coords.append(ca)
    
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

    CA_coords = np.array(CA_coords)
    res_label = np.array(res_label).reshape(len(res_label),1)
    chi_angles = np.array(chi_angles)

    return CA_coords, res_label, chi_angles
    
        
                
                    


        




if __name__ == "__main__":
    train_pdbs, train_pdb_chains = read_domain_ids_per_chain_from_txt("./data/train_domains.txt")
    test_pdbs, test_pdb_chains = read_domain_ids_per_chain_from_txt("./data/test_domains.txt")
    #for pdb in train_pdbs:
        #download_pdb(pdb)
    #for pdb in test_pdbs:
        #download_pdb(pdb)