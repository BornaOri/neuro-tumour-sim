###                  ###
###~~~CELL NETWORK~~~###
###                  ###
from neuron import h, gui
import matplotlib.pyplot as plt
import random
import numpy as np 
import time
from neuron_models import create_simple_hh_cell

h.load_file("stdrun.hoc")

#~~~ NETWORK PARAMS ~~~#
num_E = 80 #no. of excitatory
num_I = 20 #no. of inhibitory
total_cells = num_E + num_I

#~~~ SIMULTATION PARAMS ~~~#
sim_duration = 500 #ms
h.dt = 0.025 #time step integration
v_init_global = -65 #mV (initial voltage)

#~~~ CONNECTIVITY PARAMS ~~~#
connection_probability = 0.1 #chance of connection between 2 cells
netcon_delay = 1.5 #synaptic delay

#~~~ WEIGHT PARAMS ~~~# TO BE ADJUSTED!
weight_EE = 0.001
weight_EI = 0.001
weight_IE = 0.004
weight_II = 0.004

#~~~ DRIVE PARAMS ~~~# TO BE ADJUSTED!
drive_rate = 15 #Hz, approx firing rate of background input p/cell
drive_weight = 0.001 #uS (strength of background input)


#~~~ CREATE CELL POP ~~~#
print("Creating cell pop")
cells = []
for i in range(total_cells):
    cells.append(create_simple_hh_cell(i))

E_cells = cells[:num_E] 
I_cells = cells[num_E:]
print(f"Created {len(E_cells)} E cells and {len(I_cells)} I cells")

#~~~ Adding synapse objects to cells ~~~#
print("Adding synapse objects to cells")
synapses_E = []
synapses_I = []
"""Iterate through every soma and attach 2 objects at midpoint"""
for cell_soma in cells:
    #EXCITATORY SYNAPSE (AMPA model)
    syn_E = h.Exp2Syn(cell_soma(0.5))
    syn_E.tau1 = 0.2 #rise time
    syn_E.tau2 = 2.0 #decay time
    syn_E.e = 0 #reversal potential (excitatory)
    synapses_E.append(syn_E)

    #INHIBITORY SYNAPSE (GABAa model)
    syn_I = h.Exp2Syn(cell_soma(0.5))
    syn_I.tau1 = 0.5
    syn_I.tau2 = 5.0
    syn_I.e = -75 #reversal potential (inhibitory)
    synapses_I.append(syn_I)
