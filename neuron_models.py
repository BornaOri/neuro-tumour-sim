#neuron_models
from neuron import h

def create_simple_hh_cell(cell_id):
    """ Creates single compartment neuron w/ HH model"""
    soma = h.Section(name=f"soma_{cell_id}")
    soma.L = 20
    soma.diam = 20
    soma.nseg = 1
    soma.insert("hh")
    return soma 