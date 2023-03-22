from MDsimulation import *
from tqdm import tqdm
import logging
import time

#Logging for Debugging and Terminal Messages
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.DEBUG) #INFO/DEBUGGING/etc.


#Initializing Simulation  
sim = MDSimulation(steps = 10, dt = 0.004, r_cutoff= 2.5 ,thermostat = False, kT = .1, gamma = .1, linked_cell_method = True)
sim.position_init(lattice_structure = "FCC", lattice_constant = 1.5 ,side_copies = 6)
sim.velocity_init(kT = .1)

sim.current_force = sim.force_function(0)
#Main Loop
for step in tqdm(range(sim.steps)):
    sim.velocity_verlet_step(step)
    sim.potential_energies[step+1] = sim.compute_pe(step+1)
    sim.kinetic_energies[step+1] = sim.compute_ke(step+1)

#Saving Results
logging.info("Saving Results")
#sim.xyz_output(sim.positions, "positions_linkedcell.xyz")
#sim.xyz_output(sim.velocities, "velocities_linkedcell.xyz")
#np.save('Results/kT3_velocities', sim.velocities)
#np.save('Results/test_pos_NT_WL.npy',sim.positions)
#np.save('Results/linkcell_test_link.npy', sim.linked_cell.link)
#np.save('Results/linkcell_test_header.npy', sim.linked_cell.header)
#np.save('Results/test_kin_NT_WL.npy',sim.kinetic_energies)
#np.save('Results/test_pot_NT_WL.npy',sim.potential_energies)
logging.info("The simulation ended successfully!")