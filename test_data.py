from utils.data import *


def test_get_Backbone_atom_coords():
    pdb_chain = ("5ea0", "H")
    atom_coords, res_label = get_Backbone_atom_coords(pdb_chain)
    assert res_label.shape == (len(res_label),1)
    assert atom_coords.shape == (len(res_label), 3, 3)





