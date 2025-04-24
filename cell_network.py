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
weight_EE = 0.003
weight_EI = 0.001
weight_IE = 0.01
weight_II = 0.01

#~~~ DRIVE PARAMS ~~~# TO BE ADJUSTED!
drive_rate = 15 #Hz, approx firing rate of background input p/cell
drive_weight = 0.01 #uS (strength of background input)


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

print(f"DEBUG: Sample Synapse Params:")
print(f" Synapse E[0]: tau1={synapses_E[0].tau1}, tau2={synapses_E[0].tau2}, e={synapses_E[0].e}")
print(f" Synapse I[0]: tau1={synapses_I[0].tau1}, tau2={synapses_I[0].tau2}, e={synapses_I[0].e}")
print(f" Synapse E[99]: tau1={synapses_E[99].tau1}, tau2={synapses_E[99].tau2}, e={synapses_E[99].e}")
print(f" Synapse I[99]: tau1={synapses_I[99].tau1}, tau2={synapses_I[99].tau2}, e={synapses_I[99].e}")

#~~~IMPLEMENTING NETWORK CONNECTIVITY~~~#
netcons = [] #will store all connection objects
spike_threshold = -20 #Threshold to count as a spike in mV (voltage)

for i, pre_cell_soma in enumerate(cells):
    is_presyn_E =  (i < num_E) #determines if presynaptic is excitatory or inhibitory
    for j, post_cell_soma in enumerate(cells):
        if i == j:
            continue #Prevents neuron from connecting to itself
        if random.random() < connection_probability:
            #Decide if inhib/excitatory and strength/weight to use
            if is_presyn_E: 
                target_syn = synapses_E[j] #Activate E_synapse on reciever (j)
                #check if reciever j is E or I to pick right weight
                weight = weight_EE if (j < num_E) else weight_EI
            else: #if sendor inhibitory
                target_syn = synapses_I[j] #Active I_synapse on reciever (j)
                weight = weight_IE if (j < num_I) else weight_II
            #what voltage to watch on sender
            source_v = pre_cell_soma(0.5)._ref_v #Monitor voltage in the middle of sender soma 
            #create connection object, source_v & target_syn linked, sec tells NetCon which specific cell part to watch for threshold
            nc = h.NetCon(source_v, target_syn, sec = pre_cell_soma)
            #~~~ SET CONNECTION PROPERTIES ~~~#
            nc.threshold = spike_threshold #Voltage needed for sender to trigger netcon
            nc.delay = netcon_delay #time for signal to travel (ms)
            nc.weight[0] = weight #how strong connection is

            netcons.append(nc) # Keep track of connection
print(f"Created {len(netcons)} random connections")

print(f"DEBUG: Checking first 15 NetCon weights:")
for k, nc in enumerate(netcons):
    if k >= 15: break # Only print the first few
    # We need to figure out source/target types to know expected weight
    # This is tricky without storing more info, but let's just print the weight
    print(f"  NetCon {k}: weight[0] = {nc.weight[0]:.4f}")

#~~~SIMULATE BACKGROUND NOISE~~~#
#give each neuron some random excitatory inputs

netstims = [] #hold spike generator objects
drive_netcons = [] #hold connections from generators to neurons

drive_interval = 1000.0 / drive_rate #avg. time between drive spikes from rate

for j, target_syn in enumerate(synapses_E):
    stim = h.NetStim() #neurons spike generator
    #configure generator
    stim.interval = drive_interval #avg. time between spikes (control rate)
    stim.number = 1e9 #generates spikes for whole sim
    stim.noise = 1.0 #make timing somewhat random
    stim.start = 50 #start generating spikes after 50ms
    netstims.append(stim) # explicitly store the NetStim object

    #~~~CONNECT THE GENERATOR TO THE CELL'S E SYNAPSE~~~#
    nc_drive = h.NetCon(stim, target_syn)
    nc_drive.delay = 0.1 #ms (short delay)
    nc_drive.weight[0] = drive_weight #strength of each drive spike (TO BE ADJUSTED)

    drive_netcons.append(nc_drive) #stores connections

print(f"Added background noise to {len(drive_netcons)} cells.")

print(f"DEBUG: Checking first 5 Drive NetCon weights:")
for k, nc_drive in enumerate(drive_netcons):
    if k >= 5: break
    print(f"Drive NetCon {k}: weight[0] = {nc_drive.weight[0]:.4f}")

#~~~SPIKE RECORDING SETUPS~~~#
spike_times_vec = h.Vector() #for spike times
spike_ids_vec = h.Vector() #ID of neuron that spiked

spike_recorders = [] #maybe to be used to track netcons

for i, cell_soma in enumerate(cells):
    #create netcon to record cells spikes
    nc_record = h.NetCon(cell_soma(0.5)._ref_v, None, sec = cell_soma)
    nc_record.threshold = spike_threshold #same threshold as before
    #when spike detected past threshold, add time to spike_times_vec
    nc_record.record(spike_times_vec, spike_ids_vec, i)
    spike_recorders.append(nc_record)


record_voltage_indicies = [0, 40, 80, 99]
v_recorders = {} #dictionary to store voltage vecs
t_vec = h.Vector().record(h._ref_t)

for i in record_voltage_indicies:
    cell_soma = cells[i]
    v_vec = h.Vector().record(cell_soma(0.5)._ref_v) #rec V
    v_recorders[i] = v_vec

#~~~RUN SIMULATION~~~#
h.tstop = sim_duration #set simulation time
h.v_init = v_init_global #set starting voltage for all neruons
print(f"Setting starting voltage at {h.v_init} mV")

h.stdinit() #initialise all states

print(f"Running simulation for {sim_duration} ms")
t_start = time.time() #record real world time finishing
h.run() 
t_end = time.time() #record real world time ending
print(f"Simulation finished in {t_end - t_start:.2f} seconds")

#~~~VISUALISE & SAVE RESULTS~~~#
plt.figure(figsize=(12, 7)) #creates figure window for plot
spike_times = np.array(spike_times_vec)
py_spike_ids = [float(id_val) for id_val in spike_ids_vec]
spike_ids = np.array(py_spike_ids)

###debug length check
print(f"DEBUG: Length of spike_times_vec (Hoc): {int(spike_times_vec.size())}")
print(f"DEBUG: Length of spike_ids_vec (Hoc):   {int(spike_ids_vec.size())}")
print(f"DEBUG: Length of spike_times (NumPy): {len(spike_times)}")
print(f"DEBUG: Length of spike_ids (NumPy):   {len(spike_ids)}")
if spike_ids.size > 0:
    print(f"DEBUG: Unique Neuron IDs that spiked: {np.unique(spike_ids)}")
###
plt.scatter(spike_times, spike_ids, marker='.', s=5, c='black') #creates scatter plot
#labels
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")
plt.title(f"Network activity (simple HH, N={total_cells})")
#set limits of axes to match sim
plt.xlim(0, sim_duration)
plt.ylim(-1, total_cells)

#save plot to PNG with unique timestamp as name
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_filename = f"network_raster_simple_{timestamp}.png"
plt.savefig(output_filename)
print(f"RASTER PLOT SAVED TO: {output_filename}")

#membrane potential plot testing
print("Plotting voltage traces...")
plt.figure(figsize=(12, 5))
for i, v_vec in v_recorders.items():
    cell_type = "E" if i < num_E else "I"
    plt.plot(t_vec, v_vec, label=f'Cell {i} ({cell_type})')

plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Sample Neuron Voltage Traces")
plt.legend()
plt.grid(True)

# Save voltage plot
timestamp = time.strftime("%Y%m%d-%H%M%S")
voltage_filename = f'network_voltages_{timestamp}.png'
plt.savefig(voltage_filename)
print(f"--- VOLTAGE PLOT SAVED TO: {voltage_filename} ---")