
from neuron import h, gui
import matplotlib.pyplot as plt
import time

h.load_file("stdrun.hoc")

#Defines the Neuron's Morphology
soma = h.Section(name = "soma")
soma.L = 20 #length in microns
soma.diam = 20 #diameter in microns
#soma.nseg = 1 #n. of segments

soma.insert("hh") #Uses Hodgkin-Huxley ion channel mechanism

#Stimulus/Current Injection
iclamp = h.IClamp(soma(0.5)) #Clamp at mid soma (0.5)

iclamp.delay = 5 #Time pulse starts in ms
iclamp.dur = 1 #Duration of pulse in ms
iclamp.amp = 0.3 #Amplitude- to be changed for spike

#Recording Variables
v_vec = h.Vector().record(soma(0.5)._ref_v) #Rec voltage @ middle of soma
t_vec = h.Vector().record(h._ref_t) #Rec time
initial_voltage_setter = -80
#Simulation Params
h.tstop = 40 #How long simulation runs (ms)
h.v_init = initial_voltage_setter
#h.finitialize(0) #Initial resting potential & calculate channel states
h.stdinit()
print(f"{soma(0.5).v:.4f}")

h.run()
print(f"First 5 voltages from v_vec: {list(v_vec)[:5]}")
#Plotting Results using matplotlib
plt.figure(figsize=(8,4))
plt.plot(t_vec, v_vec)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("so not a sigma way to debug rn")

#mod png
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_filename = f"spike_plot_{timestamp}.jpeg"

plt.savefig(output_filename) #makes png of plot

print(f"Plot saved to {output_filename}")




