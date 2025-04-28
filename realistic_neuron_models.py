from neuron import h
#import lfpykit
from LFPy import Cell as LFPyCell # Correct import
import numpy as np
import random
import os

# --- REMOVE THIS LINE - Causes Circular Import ---
# from realistic_neuron_models import L23PyramidalCell, L23BasketCell # <<< DELETE THIS LINE >>>
# --- END REMOVAL ---

# Optional: Load compiled mechanisms explicitly
# ... (rest of optional loading code remains the same) ...

#Base class for basic neuron morphology using lfpykit
class RealisticNeuronTemplate:
    """Base class using LFPy to handle morphology and basic setup.""" # User comment retained
    def __init__(self, cell_id, swc_file, Ra=150.0, cm=1.0, v_init=-65.0, nsegs_method='lambda_f', lambda_f=100):
        self.cell_id = cell_id
        self.swc_file = swc_file
        self.Ra = Ra
        self.cm = cm
        self.v_init = v_init
        print(f"Initializing RealisticNeuronTemplate {self.cell_id} from {self.swc_file}...")

        Rm = 30000.0
        self.g_pas = 1.0 / Rm
        self.e_pas = v_init

        self.gnabar_hh = 0.120
        self.gkbar_hh = 0.036
        self.gl_hh = 0.0 # Assuming 'pas' handles all leak
        self.ena = 50
        self.ek = -77

        self.cell_parameters = {
            'morphology': self.swc_file,
            'v_init': v_init,
            'passive': False,
            'Ra': Ra,
            'cm': cm,
            'nsegs_method': nsegs_method,
            'lambda_f': lambda_f,
            'delete_sections': False # Keep this False to access sections
        }
        print(f"  Loading Cell {self.cell_id} morphology using LFPy...")
        self.cell = LFPyCell(**self.cell_parameters)

        self.all_sections = list(self.cell.allseclist)
        print(f"  Successfully loaded {len(self.all_sections)} sections (from Python list).")

        self.soma = None
        self.dendrites = []
        self.axon = []

        if not self.all_sections:
             print("  WARNING: No sections loaded, cannot identify parts.")
        else:
            max_diam = 0
            potential_soma = None
            for sec in self.all_sections:
                current_diam = sec.diam
                if current_diam > max_diam:
                    max_diam = current_diam
                    potential_soma = sec
            if potential_soma:
                 self.soma = potential_soma
                 print(f"  Identified soma (heuristic): {self.soma.name()} with diameter {max_diam:.2f}")
            else:
                 print(f"  WARNING: Could not identify soma using diameter heuristic.")

            for sec in self.all_sections:
                if sec == self.soma: continue
                if 'axon' in sec.name().lower():
                     self.axon.append(sec)
                else:
                     self.dendrites.append(sec)
            print(f"  Identified {len(self.dendrites)} dendrite sections and {len(self.axon)} axon sections (heuristic).")

        if not self.soma: print(f"  WARNING: Soma section not successfully identified for cell {self.cell_id}!")

        self._assign_biophysics()

        self.syn_E_list = []
        self.syn_I_list = []
        self._add_synapse_placeholders()

    def _assign_biophysics(self):
        """Insert mechanisms and set parameters for all sections.""" # Retained docstring
        print(f"  Assigning biophysics to Cell {self.cell_id}...")
        if not self.all_sections:
            print("    No sections found in list, cannot assign biophysics.")
            return

        num_inserted_pas = 0
        num_inserted_hh = 0
        for sec in self.all_sections:
            if h.name_declared('pas'):
                sec.insert('pas')
                sec.g_pas = self.g_pas
                sec.e_pas = self.e_pas
                num_inserted_pas += 1
            else:
                 print("    WARNING: 'pas' mechanism not found/loaded. Leak current might be missing or rely solely on hh's gl_hh.")

            if h.name_declared('hh'):
                sec.insert('hh')
                sec.gnabar_hh = self.gnabar_hh
                sec.gkbar_hh = self.gkbar_hh
                sec.gl_hh = self.gl_hh
                sec.ena = self.ena
                sec.ek = self.ek
                num_inserted_hh += 1
            else:
                print("    ERROR: 'hh' mechanism not found/loaded. Cannot simulate spiking!")

        print(f"    Inserted 'pas' into {num_inserted_pas} sections.")
        print(f"    Inserted 'hh' into {num_inserted_hh} sections.")


    def _add_synapse_placeholders(self, num_syn_each_type=10):
        """Adds placeholder synapses to somewhat realistic locations.""" # User comment retained
        print(f"  Adding {num_syn_each_type} E and I synapse placeholders to Cell {self.cell_id}...")
        if not self.all_sections:
             print("    No sections available for synapse placement.")
             return

        possible_e_targets = self.dendrites if self.dendrites else ([self.soma] if self.soma else [])
        possible_i_targets = ([self.soma] if self.soma else []) + self.dendrites[:min(len(self.dendrites), 5)]
        if not possible_i_targets and possible_e_targets: possible_i_targets = possible_e_targets
        if not possible_e_targets and not possible_i_targets:
             print("    No suitable target sections (soma/dendrites) found for synapses.")
             return

        for _ in range(num_syn_each_type):
            if possible_e_targets:
                target_sec_e = random.choice(possible_e_targets)
                loc = random.random()
                syn_e = h.Exp2Syn(target_sec_e(loc))
                syn_e.tau1, syn_e.tau2, syn_e.e = 0.2, 2.0, 0
                self.syn_E_list.append(syn_e)
            if possible_i_targets:
                 target_sec_i = random.choice(possible_i_targets)
                 loc = random.random()
                 syn_i = h.Exp2Syn(target_sec_i(loc))
                 syn_i.tau1, syn_i.tau2, syn_i.e = 0.5, 5.0, -75
                 self.syn_I_list.append(syn_i)
        print(f"    Finished adding synapse placeholders.")

# --- Specific Cell Type Classes ---

class L23PyramidalCell(RealisticNeuronTemplate):
    """ Inherits from base, adds Pyramidal specific channels like Im """ # User comment retained
    def __init__(self, cell_id, swc_file, **kwargs):
        super().__init__(cell_id, swc_file, **kwargs)
        print(f"  Applying L2/3 Pyramidal specifics to Cell {self.cell_id}...")

        gbar_im_initial_guess = 0.0001
        im_mech_name = 'Im'

        if h.name_declared(im_mech_name):
            target_sections = ([self.soma] if self.soma else []) + self.dendrites[:min(len(self.dendrites), 10)]
            if not target_sections and self.all_sections: target_sections = self.all_sections
            count = 0
            for sec in target_sections:
                 if not hasattr(sec, 'gbar_Im'):
                     sec.insert(im_mech_name)

                 im_param_name = 'gbar_Im' # Assumed parameter name

                 try:
                     setattr(sec, im_param_name, gbar_im_initial_guess)
                     count += 1
                 except AttributeError:
                      try:
                          im_param_name_alt = 'gbar' # Fallback attempt
                          setattr(sec, im_param_name_alt, gbar_im_initial_guess)
                          count += 1
                          print(f"    Note: Used '{im_param_name_alt}' instead of '{im_param_name}' for {sec.name()}")
                      except AttributeError:
                           print(f"    WARNING: Could not set '{im_param_name}' or '{im_param_name_alt}'. Check parameter name in {im_mech_name}.mod")

            print(f"    Applied '{im_mech_name}' to {count} sections with initial gbar = {gbar_im_initial_guess:.5f}")
        else:
            print(f"    WARNING: '{im_mech_name}' mechanism not found/loaded. Cannot insert.")

class L23BasketCell(RealisticNeuronTemplate):
    """ Inherits from base, adds Basket cell specific channels like Kv3 """ # User comment retained
    def __init__(self, cell_id, swc_file, **kwargs):
        super().__init__(cell_id, swc_file, **kwargs)
        print(f"  Applying Basket Cell specifics to Cell {self.cell_id}...")

        kv3_mech_name = 'Kv3_1' # Based on your .mod file
        gbar_kv3_initial_guess = 0.01

        if h.name_declared(kv3_mech_name):
            target_sections = ([self.soma] if self.soma else [])
            if not target_sections and self.all_sections: target_sections = self.all_sections
            count = 0
            for sec in target_sections:
               kv3_param_name = 'gbar_Kv3_1' # Assumed convention gbar_SUFFIX

               if not hasattr(sec, kv3_param_name):
                   sec.insert(kv3_mech_name)
               try:
                   setattr(sec, kv3_param_name, gbar_kv3_initial_guess)
               except AttributeError:
                    try:
                        kv3_param_name_alt = 'gbar' # Fallback based on your .mod PARAMETER block
                        setattr(sec, kv3_param_name_alt, gbar_kv3_initial_guess)
                        print(f"    Note: Used '{kv3_param_name_alt}' instead of '{kv3_param_name}' for {sec.name()}")
                    except AttributeError:
                         print(f"    WARNING: Could not set '{kv3_param_name}' or '{kv3_param_name_alt}'. Check parameter name in {kv3_mech_name}.mod")
               count += 1
            print(f"    Applied '{kv3_mech_name}' to {count} sections with initial gbar = {gbar_kv3_initial_guess:.4f}")
        else:
            print(f"    WARNING: '{kv3_mech_name}' mechanism not found/loaded.")

