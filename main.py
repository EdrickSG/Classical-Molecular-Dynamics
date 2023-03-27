from MDsimulation import *
from tqdm import tqdm
import logging
import sys

#Logging for Debugging and Terminal Messages
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.DEBUG) #INFO/DEBUGGING/etc.

#Loading velocities
#initial_velocities = np.load('250_p_kT_1.npy')

#Initializing Simulation  
sim = MDSimulation(steps = 5000, dt = 0.004, r_cutoff= 2.5 ,thermostat = False, kT = .1, gamma = .1, linked_cell_method = True) 
sim.position_init(lattice_structure = "SC", lattice_constant = 1.058078 ,side_copies = 10) # lattice_structure ="FCC" and lattice_constant = 1.5(471) for solid
sim.velocity_init( kT = .728)

#np.save('686_kT_10.npy',sim.current_velocities)

# To save positions (to delete)
positions = np.zeros((sim.steps+1,sim.num_particles,sim.dim))
positions[0] = sim.current_positions
velocities = np.zeros((sim.steps+1,sim.num_particles,sim.dim))
velocities[0] = sim.current_velocities

#Main Loop
for step in tqdm(range(sim.steps)):
    sim.velocity_verlet_step()
    #Saving observables
    sim.potential_energies[step+1] = sim.current_potential
    sim.kinetic_energies[step+1] = sim.compute_ke()
    positions[step+1] = sim.current_positions
    velocities[step+1] = sim.current_velocities

#Saving Results
logging.info("Saving Results")
#sim.xyz_output(sim.positions, "positions_linkedcell.xyz")
sim.xyz_output(velocities, "velocity.out")
#np.save('Results/kT3_velocities', velocities)
np.save('Results/test_pos.npy',positions)
#np.save('Results/linkcell_test_link.npy', sim.linked_cell.link)
#np.save('Results/linkcell_test_header.npy', sim.linked_cell.header)
np.save('Results/kinetic.npy',sim.kinetic_energies)
np.save('Results/potential.npy',sim.potential_energies)
logging.info("The simulation ended successfully!")