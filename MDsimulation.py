import numpy as np
import copy as cp
import itertools as it
import logging
from Thermostat import *
from LinkedCell import *


class MDSimulation:

    def __init__(self, steps = 5000, dt = 0.004 ,box_len = 10, r_cutoff= 2.5 ,thermostat = False, kT = 1, gamma = 0.01):
        self.steps = steps
        self.dt = dt
        self.box_len = box_len
        self.r_cutoff= 2.5
        self.kinetic_energies = np.zeros((self.steps+1))
        self.potential_energies = np.zeros((self.steps+1))
        self.dim = None         #Currently all the simulations are in 3D...
        self.num_particles = None #Once the num_particles is determined, we set the following arrays
        self.positions = None #np.zeros((self.steps+1,self.num_particles,self.dim))
        self.velocities = None #np.zeros((self.steps+1,self.num_particles,self.dim))
        logging.info(f'Simulation created with {self.steps} steps of length dt = {self.dt} and box length equal to {self.box_len}')
        self.thermostat = Thermostat(self.dt, active = thermostat, kT = kT,  gamma = gamma)
        self.linked_cell = None
        
    def position_init(self, lattice_structure = "FCC" , side_copies = 2 ):

        if lattice_structure == "FCC":
            self.dim = 3
            unit_cell= np.array([[0, 0, 0],[0.5, 0.5, 0],[0, 0.5, 0.5],[0.5, 0, 0.5]])
            initial_positions = self.generate_initial_positions(unit_cell,side_copies)
            self.num_particles = len(initial_positions)
            #if self.num_particles != (side_copies+1)**3+side_copies**2*(side_copies+1)*3: # ToDo: Compute again the number of atoms
                #logging.error(f'The number of atoms is not consistent! :(')
        elif lattice_structure == "BCC":
            self.dim = 3
            unit_cell = np.array([[0, 0, 0], [.5,.5,.5]])
            initial_positions = self.generate_initial_positions(unit_cell,side_copies)
            self.num_particles = len(initial_positions)  #side_copies**3* 2->(len(unit_cell))
        else:
                logging.error(f'The argument lattice_structure can be either FCC or BCC!')
        #Initializing Positions and Setting Initial Position and Potential Energy
        self.positions = np.zeros((self.steps+1,self.num_particles,self.dim))
        self.positions[0] = initial_positions
        self.potential_energies[0] = self.compute_pe(0)
        logging.info(f'{self.num_particles} particles were placed on {lattice_structure} lattice structure ({side_copies} unit cells per side)')
    
    def generate_initial_positions(self, unit_cell, side_copies):
        displacements = np.array([np.array([x,y,z]) for x in range(side_copies) for y in range(side_copies) for z in range(side_copies)  ])
        all_points = unit_cell
        for i in range(len(displacements)):
            all_points = np.concatenate((all_points, unit_cell+displacements[i]))
        shift = 0 # self.box_len/(((side_copies*2+2)*2))
        scale = (self.box_len-2*shift)/side_copies
        all_points = np.unique(all_points, axis=0)  #To delete
        all_points = all_points*scale+np.array([shift,shift,shift])
        return all_points

    def velocity_init(self, kT):
        self.velocities = np.zeros((self.steps+1,self.num_particles,self.dim))
        self.velocities[0] = np.random.normal(loc=0,scale=np.sqrt(kT) ,size=(self.num_particles ,self.dim))
        self.kinetic_energies = np.zeros((self.steps+1))
        self.kinetic_energies[0] = self.compute_ke(0)
        logging.info(f'Temperature for initial velocities was kT={kT}')
    
    def linked_cell_init(self, cell_len):
        if cell_len<self.r_cutoff:
            logging.error(f"The cell length of the linked cell method ({cell_len}) should larger than the cut-off radio ({self.r_cutoff}). See main.py")
            exit()
        self.linked_cell = LinkedCell(cell_len, self.box_len, self.num_particles)
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
            f_mag = 24 * (2*((1/r_mag)**14) - (1/r_mag)**8)
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
    
    def compute_forces(self, step):
        return np.array([self.lj_force(p,step) for p in range(self.num_particles)])
    
    def simple_lj_force_pair(self,p1, p2, step):  #The only difference with lj_force_pair is that we do not use the minimum image convention
        r = self.positions[step][int(p1)] - self.positions[step][int(p2)] # Todo: Make p1 and p2 integer from the beginnign
        r_mag = np.linalg.norm(r)
        if r_mag <= self.r_cutoff:
            f_mag = 24 * (2*((1/r_mag)**14) - (1/r_mag)**8)
        else:
            f_mag = 0
        return f_mag * r
    
    def compute_forces_linkedcell(self, step):
        force = np.zeros((self.num_particles,self.dim))
        side_cells = self.linked_cell.side_cells
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
                            #if (pn == central_cell_particles[p1]):
                            #    continue
                            pair_force = self.lj_force_pair(central_cell_particles[p1], pn, step)
                            force[central_cell_particles[p1]] += pair_force
                            force[pn] -= pair_force
        return force
    
    def velocity_verlet_step(self, step):
        #force = self.compute_forces(step)
        force = self.compute_forces_linkedcell(step)
        self.positions[step+1] = self.positions[step] + self.velocities[step]*self.dt +0.5*force*(self.dt**2)
        self.positions[step+1] += self.thermostat.positions(self.velocities[step])   #Adding the contribution of the thermostat
        #Boundary conditions
        for k in range(self.num_particles):
            for i in range(self.dim):
                if self.positions[step+1][k][i]>self.box_len:
                    self.positions[step+1][k][i]= self.positions[step+1][k][i] - self.box_len
                if self.positions[step+1][k][i]<0:
                    self.positions[step+1][k][i] = self.positions[step+1][k][i] + self.box_len
        #force_next = self.compute_forces(step+1)
        force_next = self.compute_forces_linkedcell(step+1)
        self.velocities[step+1] = self.velocities[step] + 0.5*(force+force_next)*(self.dt)
        self.velocities[step+1] += self.thermostat.velocities(force)   #Adding the contribution of the thermostat
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
        return 4*((1/r_mag)**12- (1/r_mag)**6)
    
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
