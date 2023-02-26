from data.data import *

def test_secondary_structure_length():
    test_pdb_pair = ('1bur','S')
    amino_acid, secondary_structure = get_secondary_structure(test_pdb_pair, data_dir="./pdb")
    assert len(amino_acid) == len(secondary_structure)

def test_resiude_length():
    test_pdb_pair = ('1bur','S')
    amino_acid, _ = get_secondary_structure(test_pdb_pair, data_dir="./pdb")
    _, res_label, _ = get_pdb_data(test_pdb_pair)
    assert len(res_label) == len(amino_acid)

def test_secondary_structure_code():
    test_pdb_pair = ('1bur','S')
    _, secondary_structure = get_secondary_structure(test_pdb_pair, data_dir="./pdb")
    for ss in secondary_structure:
        assert ss == "H" or ss == "E" or ss == "L"


