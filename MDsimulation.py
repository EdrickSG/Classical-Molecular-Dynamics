import numpy as np
import copy as cp
import itertools as it
import logging

class MDSimulation:

    def __init__(self, steps = 5000, dt = 0.004 ,box_len = 10):
        self.steps = steps
        self.dt = dt
        self.box_len = box_len
        self.kinetic_energies = np.zeros((self.steps+1))
        self.potential_energies = np.zeros((self.steps+1))
        self.second_potential = np.zeros((self.steps+1)) #To compare with potential_energy
        self.dim = None         #Currently all the simulations are in 3D...
        self.num_particles = None #Once the num_particles is determined, we set the following arrays
        self.positions = None #np.zeros((self.steps+1,self.num_particles,self.dim))
        self.velocities = None #np.zeros((self.steps+1,self.num_particles,self.dim))
        logging.info(f'Simulation created with {steps} steps of length dt = {dt} and box length equals to {box_len}')
        
    def position_init(self, lattice_structure = "FCC" , side_copies = 2 ):
        if lattice_structure == "BCC":
            self.dim = 3
            unit_cell = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [.5,.5,.5]])
            initial_positions = self.generate_initial_positions(unit_cell,side_copies)
            self.num_particles = len(initial_positions)
            if self.num_particles != (side_copies+1)**3+side_copies**3:
                logging.error(f'The number of atoms is not consistent! :(')
        elif lattice_structure == "FCC":
            self.dim = 3
            unit_cell= np.array([[0, 0, 0],[1, 0, 0],[1, 1, 0],[0, 1, 0],[0, 0, 1],[1, 0, 1],
                                 [1, 1, 1],[0, 1, 1],[0.5, 0.5, 0],[0, 0.5, 0.5],[0.5, 0, 0.5],
                                 [0.5, 0.5, 1],[1, 0.5, 0.5],[0.5, 1, 0.5]])
            initial_positions = self.generate_initial_positions(unit_cell,side_copies)
            self.num_particles = len(initial_positions)
            if self.num_particles != (side_copies+1)**3+side_copies**2*(side_copies+1)*3:
                logging.error(f'The number of atoms is not consistent! :(')
        else:
                logging.error(f'The argument lattice_structure can be either FCC or BCC!')
        #Initializing Positions and Setting Initial Position and Potential Energy
        self.positions = np.zeros((self.steps+1,self.num_particles,self.dim))
        self.positions[0] = initial_positions
        self.potential_energies[0] = self.compute_pe(0)
        self.second_potential[0] = self.compute_pe(0) 
        logging.info(f'{self.num_particles} particles were placed on {lattice_structure} lattice structure ({side_copies} unit cells per side)')
    
    def generate_initial_positions(self, unit_cell, side_copies):
        displacements = np.array([np.array([x,y,z]) for x in range(side_copies) for y in range(side_copies) for z in range(side_copies)  ])
        all_points = unit_cell
        for i in range(len(displacements)):
            all_points = np.concatenate((all_points, unit_cell+displacements[i]))
        shift =  self.box_len/(((side_copies*2+2)*2))
        scale = (self.box_len-2*shift)/side_copies
        all_points = np.unique(all_points, axis=0)
        all_points = all_points*scale+np.array([shift,shift,shift])
        return all_points

    def velocity_init(self):
        self.velocities = np.zeros((self.steps+1,self.num_particles,self.dim))
        self.velocities[0] = np.random.normal(loc=0,size=(self.num_particles, self.dim))
        self.kinetic_energies = np.zeros((self.steps+1))
        self.kinetic_energies[0] = self.compute_ke(0)

    def lj_force_pair(self,p1, p2, step):
        r = self.positions[step][p1] - self.positions[step][p2]
        #Minimum Image Convention
        for i in range(self.dim):
            if r[i]>self.box_len/2:
                r[i]=r[i] - self.box_len
            if r[i]<-self.box_len/2:
                r[i]=r[i] + self.box_len
        r_mag = np.linalg.norm(r)
        f_mag = 24 * (2*((1/r_mag)**14) - (1/r_mag)**8)
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
    
    def velocity_verlet_step(self, step):
        force = self.compute_forces(step)
        self.second_potential[step+1] = self.compute_pe(step)
        self.positions[step+1] = self.positions[step] + self.velocities[step]*self.dt +0.5*force*(self.dt**2)
        #Boundary conditions
        for k in range(self.num_particles):
            for i in range(self.dim):
                if self.positions[step+1][k][i]>self.box_len:
                    self.positions[step+1][k][i]= self.positions[step+1][k][i] - self.box_len
                if self.positions[step+1][k][i]<0:
                    self.positions[step+1][k][i] = self.positions[step+1][k][i] + self.box_len
        force_next = self.compute_forces(step+1)
        self.velocities[step+1] = self.velocities[step] + 0.5*(force+force_next)*(self.dt)

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
    
    def xyz_output(self):
        file = open("Results/test.xyz", "w")
        file.write("\n")
        file.write(f"{len(self.positions[0])}\n")
        file.write(f"\n")
        for step in range(len(self.positions)):
            for index, all_pos in enumerate(self.positions[step]):
                file.write(f"atom{index} {all_pos[0]} {all_pos[1]} {all_pos[2]}\n")
            file.write("\n")