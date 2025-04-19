
from neuron import h, gui
import matplotlib.pyplot as plt
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

#Simulation Params
h.tstop = 20 #How long simulation runs (ms)
h.finitialize(-65) #Initial resting potential & calculate channel states
h.run()

#Plotting Results using matplotlib
plt.figure(figsize=(8,4))
plt.plot(t_vec, v_vec)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Single HH Neuron Sim")

plt.savefig("single_neuron_spike.png") #makes png of plot

print("Plot saved to singleneuronspike png")
plt.show()


