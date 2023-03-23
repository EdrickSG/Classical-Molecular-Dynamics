from MDsimulation import *
from tqdm import tqdm
import logging
import time

#Logging for Debugging and Terminal Messages
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.DEBUG) #INFO/DEBUGGING/etc.

#Initializing Simulation  
sim = MDSimulation(steps = 500, dt = 0.004, r_cutoff= 5 ,thermostat = False, kT = .1, gamma = .1, linked_cell_method = False) 
sim.position_init(lattice_structure = "BCC", lattice_constant = 2 ,side_copies = 5) # lattice_constant = 1.5(471) for solid
sim.velocity_init(kT = 1)

# To save positions (to delete)
positions = np.zeros((sim.steps+1,sim.num_particles,sim.dim))
positions[0] = sim.current_positions

sim.current_force = sim.force_function()

#Main Loop
for step in tqdm(range(sim.steps)):
    sim.velocity_verlet_step()
    #Saving observables
    sim.potential_energies[step+1] = sim.compute_pe()
    sim.kinetic_energies[step+1] = sim.compute_ke()
    positions[step+1] = sim.current_positions

#Saving Results
logging.info("Saving Results")
#sim.xyz_output(sim.positions, "positions_linkedcell.xyz")
#sim.xyz_output(sim.velocities, "velocities_linkedcell.xyz")
#np.save('Results/kT3_velocities', velocities)
np.save('Results/test_pos_.npy',positions)
#np.save('Results/linkcell_test_link.npy', sim.linked_cell.link)
#np.save('Results/linkcell_test_header.npy', sim.linked_cell.header)
np.save('Results/test_kin.npy',sim.kinetic_energies)
np.save('Results/test_pot.npy',sim.potential_energies)
logging.info("The simulation ended successfully!")