from MDsimulation import *
from matplotlib import pyplot as plt
from tqdm import tqdm
import logging

#Logging for Debugging and Terminal Messages
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO) #INFO/DEBUGGING/etc.

#Initializing Simulation
sim=MDSimulation(steps = 1000, dt = 0.004, box_len = 10)
sim.position_init(lattice_structure = "BCC", side_copies = 2)
sim.velocity_init()

#Main Loop
for step in tqdm(range(sim.steps)):
    sim.velocity_verlet_step(step)
    sim.potential_energies[step+1] = sim.compute_pe(step+1)
    sim.kinetic_energies[step+1] = sim.compute_ke(step+1)

#Saving Results
sim.xyz_output()
np.save('testFCC_velocities', sim.velocities)
np.save('testFCC_positions',sim.positions)
np.save('testFCC_KE',sim.kinetic_energies)
np.save('testFCC_PE',sim.potential_energies)
np.save('testFCC_second_pot',sim.second_potential)