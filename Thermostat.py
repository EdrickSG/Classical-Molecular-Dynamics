import numpy as np
import logging

class Thermostat:

    def __init__(self, dt ,active = False, kT = 1, gamma = 0.01, m = 1 ):
        self.dt = dt
        self.kT = kT
        self.gamma = gamma
        self.active = active
        self.sigma = np.sqrt(2* self.kT*self.gamma/m)
        self.eta = None  # Gaussian variables
        self.xi = None
        self.An = None  #An factor in WEC paper minus force contribution
        self.previous_velocities = None
        if active:
            logging.info(f'The Vanden-Eijnden-Ciccotti thermostat is active at temperature kT = {self.kT} and friction coefficient = {self.gamma}')
        else:
            logging.info(f'The Vanden-Eijnden-Ciccotti thermostat is not active.')


    def positions(self, velocities):
        if self.active:
            self.previous_velocities = velocities
            self.eta = np.random.normal(loc=0.0,size=(len(velocities),3))
            self.xi = np.random.normal(loc=0.0,size=(len(velocities),3))
            self.An = - 0.5*(self.dt**2)*self.gamma*self.previous_velocities + self.sigma*(self.dt**(3/2))*(0.5*self.xi + 1/(2*np.sqrt(3))*self.eta )
            return  self.An
        else:
            return 0
        
    def velocities(self, force):
        if self.active:
            return - self.dt*self.gamma*self.previous_velocities + np.sqrt(self.dt)*self.sigma*self.xi - self.gamma*(0.5*force*(self.dt**2)+self.An)
        else:
            return 0
    

