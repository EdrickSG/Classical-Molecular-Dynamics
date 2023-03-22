import numpy as np
import copy as cp
import itertools as it
import logging
from Thermostat import *
from LinkedCell import *


class MDSimulation:

    def __init__(self, m = 1., sigma=1., epsilon=1., steps = 5000, dt = 0.004 , r_cutoff= 1.5 ,thermostat = False, kT = 1, gamma = 0.01, linked_cell_method = False):
        self.m = m
        self.sigma = sigma
        self.epsilon = epsilon
        self.steps = steps
        self.dt = dt
        self.box_len = None
        self.r_cutoff= r_cutoff
        self.kinetic_energies = np.zeros((self.steps+1))
        self.potential_energies = np.zeros((self.steps+1))
        self.current_force = None
        self.dim = None         #Currently all the simulations are in 3D...
        self.num_particles = None #Once the num_particles is determined, we set the following arrays
        self.positions = None #np.zeros((self.steps+1,self.num_particles,self.dim))
        self.velocities = None #np.zeros((self.steps+1,self.num_particles,self.dim))
        logging.info(f'Simulation created with {self.steps} steps of length dt = {self.dt} and cut off radius {r_cutoff}')
        self.thermostat = Thermostat(self.dt, active = thermostat, kT = kT,  gamma = gamma, m = self.m)
        self.linked_cell = None
        self.linked_cell_method_flag = linked_cell_method
        if linked_cell_method:
            self.force_function = self.compute_forces_linkedcell
        else:
            self.force_function = self.compute_forces_brutforce
            logging.info(f"The linked-cell metod is deactivate")

        
    def position_init(self, lattice_structure = "FCC" , lattice_constant = 1.5 ,side_copies = 2 ):
        if lattice_structure == "FCC":
            self.dim = 3
            unit_cell= np.array([[0, 0, 0],[0.5, 0.5, 0],[0, 0.5, 0.5],[0.5, 0, 0.5]])
            initial_positions = self.generate_initial_positions(unit_cell, side_copies, lattice_constant)
            self.num_particles = len(initial_positions)
        elif lattice_structure == "BCC":
            self.dim = 3
            unit_cell = np.array([[0, 0, 0], [.5,.5,.5]])
            initial_positions = self.generate_initial_positions(unit_cell,side_copies, lattice_constant)
            self.num_particles = len(initial_positions)  #side_copies**3* 2->(len(unit_cell))
        else:
                logging.error(f'The argument lattice_structure can be either FCC or BCC!')
        #Initializing Positions and Setting Initial Position and Potential Energy
        self.positions = np.zeros((self.steps+1,self.num_particles,self.dim))
        self.positions[0] = initial_positions
        self.box_len = side_copies*lattice_constant #initial_positions.max()
        self.potential_energies[0] = self.compute_pe(0)
        logging.info(f'{self.num_particles} particles were placed on {lattice_structure} lattice structure ({side_copies} unit cells per side) in a box of length {self.box_len}')
        if self.linked_cell_method_flag:
            self.linked_cell_init()
    
    def generate_initial_positions(self, unit_cell, side_copies, lattice_constant):
        displacements = np.array([np.array([x,y,z]) for x in range(side_copies) for y in range(side_copies) for z in range(side_copies)  ])
        initial_positions = np.array([unit_cell+displacement for displacement in displacements])
        initial_positions = initial_positions.reshape(len(unit_cell)*len(displacements), self.dim)  
        return initial_positions*lattice_constant

    def velocity_init(self, kT):
        self.velocities = np.zeros((self.steps+1,self.num_particles,self.dim))
        factor = np.sqrt(kT/self.m)
        self.velocities[0] = np.random.normal(loc=0, scale=factor, size=(self.num_particles, self.dim))
        self.kinetic_energies = np.zeros((self.steps+1))
        self.kinetic_energies[0] = self.compute_ke(0)
        logging.info(f'Temperature for initial velocities was kT={kT}')
    
    def linked_cell_init(self):
        self.linked_cell = LinkedCell(self.r_cutoff, self.box_len, self.num_particles)
        self.linked_cell.update_lists(self.positions[0])

    def lj_force_pair(self,p1, p2, step):
        r = self.positions[step][p1] - self.positions[step][p2]
        #Minimum Image Convention
        for i in range(self.dim):
            if r[i]>self.box_len/2:
                r[i]=r[i] - self.box_len
            if r[i]<-self.box_len/2:
                r[i]=r[i] + self.box_len
        r_mag = np.linalg.norm(r)
        if r_mag<= self.r_cutoff:
            f_mag = 24*(self.epsilon/self.sigma**2) * (2*((self.sigma/r_mag)**14) - (self.sigma/r_mag)**8)
        else:
            f_mag = 0
        return f_mag * r

    def lj_force(self, p, step):
        force = np.zeros(shape=self.dim)
        for part in range(self.num_particles):
            if(part == p):
                continue
            force += self.lj_force_pair(p, part, step)
        return force
    
    def compute_forces_brutforce(self, step):
        return np.array([self.lj_force(p,step) for p in range(self.num_particles)])
    
    def simple_lj_force_pair(self,p1, p2, step):  #The only difference with lj_force_pair is that we do not use the minimum image convention
        r = self.positions[step][int(p1)] - self.positions[step][int(p2)] # Todo: Make p1 and p2 integer from the beginning
        r_mag = np.linalg.norm(r)
        if r_mag <= self.r_cutoff:
            f_mag = 24*(self.epsilon/self.sigma**2) * (2*((self.sigma/r_mag)**14) - (self.sigma/r_mag)**8)
        else:
            f_mag = 0
        return f_mag * r
    
    def compute_forces_linkedcell(self, step):
        force = np.zeros((self.num_particles,self.dim))
        side_cells = self.linked_cell.side_cell_num
        for IX in range(side_cells):
            for IY in range(side_cells):
                for IZ in range(side_cells):
                    central_cell_particles, neighbor_cells_particles = self.linked_cell.interacting_particles(IX, IY, IZ)
                    num_particles_central = len(central_cell_particles)
                    for p1 in range(num_particles_central):
                        #Interaction between particles in central cell
                        for p2 in range(p1+1,num_particles_central):
                            pair_force = self.simple_lj_force_pair(central_cell_particles[p1], central_cell_particles[p2], step)  # I may use lj_force_pair...
                            force[central_cell_particles[p1]] += pair_force
                            force[central_cell_particles[p2]] -= pair_force
                        #Interaction between particles in central cell and neighbors
                        for pn in neighbor_cells_particles:
                            pair_force = self.lj_force_pair(central_cell_particles[p1], pn, step)
                            force[central_cell_particles[p1]] += pair_force
                            force[pn] -= pair_force
        return force
    
    def velocity_verlet_step(self, step):
        force = self.current_force
        self.positions[step+1] = self.positions[step] + self.velocities[step]*self.dt +0.5*force*(self.dt**2)/self.m
        self.positions[step+1] += self.thermostat.positions(self.velocities[step])   #Adding the contribution of the thermostat
        #Boundary conditions
        for k in range(self.num_particles):
            for i in range(self.dim):
                if self.positions[step+1][k][i]>self.box_len:
                    self.positions[step+1][k][i]= self.positions[step+1][k][i] - self.box_len
                if self.positions[step+1][k][i]<0:
                    self.positions[step+1][k][i] = self.positions[step+1][k][i] + self.box_len
        force_next = self.force_function(step+1)
        self.velocities[step+1] = self.velocities[step] + 0.5*(force+force_next)*(self.dt)/self.m
        self.velocities[step+1] += self.thermostat.velocities(force)   #Adding the contribution of the thermostat
        current_force = force_next
        if self.linked_cell_method_flag:
            self.linked_cell.update_lists(self.positions[step+1])

    def pe_pair(self, p1, p2, step):
        r = self.positions[step][p1] - self.positions[step][p2]
        #Minimum image convention
        for i in range(self.dim):
            if r[i]>self.box_len/2:
                r[i]=r[i] - self.box_len
            if r[i]<-self.box_len/2:
                r[i]=r[i] + self.box_len
        r_mag = np.linalg.norm(r)
        if r_mag<= self.r_cutoff:
            potential = 4*self.epsilon*((self.sigma/r_mag)**12-(self.sigma/r_mag)**6)
        else:
            potential = 0
        return potential
    
    def compute_pe(self, step):
        total_pe=0
        for i in range(self.num_particles):
            for j in range(i+1,self.num_particles):
                total_pe+=self.pe_pair(i,j,step)
        return total_pe
    
    def compute_ke(self, step):
        velocity_squared = np.array([np.dot(v,v) for v in self.velocities[step]])
        return 0.5*np.sum(velocity_squared)
    
    def xyz_output(self, particle_property, file_name): #self.positions or self.velocities
        file_complete_name = "Results/"+file_name
        file = open(file_complete_name, "w")
        for step in range(len(particle_property)):
            file.write(f"{len(particle_property[0])}\n")
            file.write("\n")
            for index, all_pos in enumerate(particle_property[step]):
                file.write(f"Ar {all_pos[0]} {all_pos[1]} {all_pos[2]}\n")

