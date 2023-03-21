from MDsimulation import *
from tqdm import tqdm
import logging
import time

#Logging for Debugging and Terminal Messages
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO) #INFO/DEBUGGING/etc.

#Initializing Simulation
sim = MDSimulation(steps = 1000, dt = 0.004, box_len = 12, thermostat = True, kT = 3, gamma = .1)
sim.position_init(lattice_structure = "FCC", side_copies = 4)
sim.velocity_init(kT = 1)
sim.linked_cell_init(cell_len = 3)

#Main Loop
for step in tqdm(range(sim.steps)):
    sim.velocity_verlet_step(step)
    sim.potential_energies[step+1] = sim.compute_pe(step+1)
    sim.kinetic_energies[step+1] = sim.compute_ke(step+1)

#Saving Results
logging.info("Saving Results")
sim.xyz_output(sim.positions, "positions.xyz")
sim.xyz_output(sim.velocities, "velocities.xyz")
#np.save('Results/kT3_velocities', sim.velocities)
np.save('Results/test_pos.npy',sim.positions)
#np.save('Results/linkcell_test_link.npy', sim.linked_cell.link)
#np.save('Results/linkcell_test_header.npy', sim.linked_cell.header)
np.save('Results/test_kin.npy',sim.kinetic_energies)
np.save('Results/test_pot.npy',sim.potential_energies)
logging.info("The simulation ended successfully!")