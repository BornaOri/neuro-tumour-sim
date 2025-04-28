#ONLY TEMPORARY FOR PYRAMIDAL AND BASKET CELLS
from neuron import h
import matplotlib.pyplot as plt
import numpy as np
import time
# Import the specific cell class you want to test
from realistic_neuron_models import L23PyramidalCell, L23BasketCell

h.load_file("stdrun.hoc")

# Parameters
CELL_TO_TEST = "Pyramidal" # Or "Basket"
if CELL_TO_TEST == "Pyramidal":
    swc_file = 'H17.06.006.11.09.04_591274508_m (1).swc' # <<< Ensure this filename is exactly correct
    CellClass = L23PyramidalCell
    cell_label = "L2/3 Pyramidal"
    iclamp_amp = 0.5
else:
    swc_file = 'Fig2b_cell1_0904091kg.CNG.swc' # <<< Ensure this filename is exactly correct
    CellClass = L23BasketCell
    cell_label = "L2/3 Basket"
    iclamp_amp = 0.8

sim_duration = 1000  # ms (Long duration to observe adaptation/sustained firing)
h.dt = 0.025         # ms
v_init = -65         # mV
h.celsius = 34       # degC

#Create Cell
cell_object = CellClass(0, swc_file, v_init=v_init)
if not cell_object.soma:
    print("Error: Soma not found in loaded model!")
    exit()
print("Cell created.")

#Stimulus current injection
iclamp = h.IClamp(cell_object.soma(0.5)) # Attach to the identified soma midpoint
iclamp.delay = 100   # ms - Wait 100ms before starting
iclamp.dur = 800     # ms - *** Make the pulse long to observe adaptation ***
iclamp.amp = iclamp_amp # Set the amplitude

#Setup recording
v_soma_vec = h.Vector().record(cell_object.soma(0.5)._ref_v) # Record soma voltage
t_vec = h.Vector().record(h._ref_t)                 # Record time points
spike_times_vec = h.Vector()
spike_threshold = -10 # Or maybe lower (-20?) - TUNE if needed
nc_record = h.NetCon(cell_object.soma(0.5)._ref_v, None, sec=cell_object.soma)
nc_record.threshold = spike_threshold
nc_record.record(spike_times_vec) # Record only times

#Run simulation
h.tstop = sim_duration
h.v_init = v_init
print(f"--- Setting h.v_init = {h.v_init} mV ---")
h.stdinit()
print(f"--- Called h.stdinit() ---")
print(f"Running simulation for {sim_duration} ms...")
t_start_sim = time.time()
h.run()
t_end_sim = time.time()
print(f"Simulation finished in {t_end_sim - t_start_sim:.2f} seconds.")

#Analyze & Plot
# --- Correction: Explicitly convert spike_times_vec before np.array ---
spike_times = np.array([]) # Initialize as empty NumPy array
conversion_successful = False
try:
    # Create a Python list by iterating through the NEURON vector and converting each element to float
    py_spike_times = [float(t) for t in spike_times_vec]
    # Create the NumPy array from the clean Python list
    spike_times = np.array(py_spike_times)
    conversion_successful = True
except Exception as e:
    print(f"ERROR during spike_times conversion to NumPy array: {e}")
# --- End Correction ---


# Calculate ISIs and Adaptation Ratio
adaptation_ratio = np.nan
first_isi = np.nan
last_isi = np.nan
print(f"  Spike Count: {len(spike_times)}")
if conversion_successful and len(spike_times) > 1: # Need at least 2 spikes for 1 ISI
    stim_start_time = iclamp.delay
    stim_end_time = iclamp.delay + iclamp.dur
    stim_spikes = spike_times[(spike_times >= stim_start_time) & (spike_times <= stim_end_time)]

    if len(stim_spikes) > 2: # Need at least 3 spikes during stim for 2 ISIs
        isis = np.diff(stim_spikes)
        first_isi = isis[0]
        last_isi = isis[-1]
        if first_isi > 1e-5: # Avoid division by zero or near-zero
             adaptation_ratio = last_isi / first_isi
        print(f"  First ISI during stim: {first_isi:.2f} ms")
        print(f"  Last ISI during stim: {last_isi:.2f} ms")
        print(f"  Adaptation Ratio (Last/First): {adaptation_ratio:.3f}")
    elif len(stim_spikes) > 1 :
         isis = np.diff(stim_spikes)
         first_isi = isis[0]
         print(f"  Only one ISI during stim: {first_isi:.2f} ms")
    else:
        print("  Not enough spikes during stimulus to calculate adaptation ratio.")
else:
     print("  Not enough spikes recorded/converted to calculate adaptation ratio.")

#Plot voltage trace
plt.figure(figsize=(12, 5))
plt.plot(t_vec, v_soma_vec, label=f'Soma Vm (Adapt Ratio: {adaptation_ratio:.3f})')
if conversion_successful and spike_times.size > 0 :
    y_min, y_max = plt.ylim()
    # Ensure y_marker is a finite number before plotting
    y_marker = y_max * 0.95 if np.isfinite(y_max) else 0
    plt.scatter(spike_times, np.full(spike_times.shape, y_marker),
                color='red', marker='|', s=100, label='Spikes')

plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title(f"Test: {cell_label} (I={iclamp.amp:.2f} nA for {iclamp.dur}ms)")
plt.legend(loc='upper right')
plt.grid(True)

# Save the plot
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_filename = f'test_{CELL_TO_TEST}_{timestamp}.png'
plt.savefig(output_filename)
print(f"--- Plot saved to {output_filename} ---")