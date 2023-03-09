from MDsimulation import *
from tqdm import tqdm
import logging

#Logging for Debugging and Terminal Messages
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.DEBUG) #INFO/DEBUGGING/etc.

#Initializing Simulation
sim = MDSimulation(steps = 5000, dt = 0.004, box_len = 10, thermostat = True, kT = 3, gamma = 0.01)
sim.position_init(lattice_structure = "BCC", side_copies = 2)
sim.velocity_init(1)

#Main Loop
for step in tqdm(range(sim.steps)):
    sim.velocity_verlet_step(step)
    sim.potential_energies[step+1] = sim.compute_pe(step+1)
    sim.kinetic_energies[step+1] = sim.compute_ke(step+1)

#Saving Results
logging.info("Saving Results")
sim.xyz_output()
np.save('Results/kT3_velocities', sim.velocities)
np.save('Results/kT3_positions',sim.positions)
np.save('Results/kT3_KE',sim.kinetic_energies)
np.save('Results/kT3_PE',sim.potential_energies)
np.save('Results/kT3_pot',sim.second_potential)
logging.info("The simulation ended successfully!")