import numpy as np
from numpy import ma
from math import pi, sin, cos
from scipy.spatial import distance

class Human: #only agent
    
    def __init__ (self, age, location, t_infected, health, quarantine, immunocompromised, t_torecuperate, chance_death_dt, freeroam, social):
        self.age = age # from 1-85
        self.location = location #X,Y position  tuple of the grid he is in
        self.t_infected = t_infected # time since infection, -1 otherwise
        self.health = health # 0 (not infected), 1(infected), 2(immune), 3(dead)
        self.quarantine = quarantine #1 if quarantined, 2 if infected but not quarantined, 0 if not quarantined because no infection
        self.immunocompromised = immunocompromised # 1 if immunocompromised, 0 if otherwise
        self.t_torecuperate = t_torecuperate # number of timesteps left to heal, -1 if not infected
        self.chance_death_dt = chance_death_dt #cahnce of dying per dt if infected, -1 if not infected
        self.freeroam = freeroam #determines if user will be affected by quarantine, self isolation or panic index (1 if not affected, 0 otherwise)
        self.social = social #determines if moves to social hours in first 8 hours. 1 if he does, 0 otherwise


    
    
    
    
    
    
    
    def movev2 (self,max_size, daily_hour, pop_density, panic_factor, centroids, grids, stay_home, stay_move_chance):
        """
        agent moves with this funciont. Movement depends on many factors, such as if the agent 
        is social, if he ignores quarantines or stay at home orders, if there is an active stay at home 
        order, and their combinations. The function updates the location of the agent and returns the new
        position

        Parameters
        ----------
        max_size : int
            max size of the map.
        daily_hour : int
            current daily hour of the simulation.
        pop_density : numpy matrix
            matrix where each value is the amount of people in that specific area.
        panic_factor : int
            panic factor of the simulation. Determines square search area
        centroids : array of doubles
            contains the centroids of all the social areas.
        grids : numpy matrix
            matrix filled with 1s or 0s, where 1s are social areas.
        stay_home : int
            if stay at home is activated, this value will be a 1.
        stay_move_chance: float
            chance of stay at home agents of moving per timestep

        Returns
        -------
        Return location of object as a tuple

        """

        
        #first 8 hours of the day, people will move to social areas and stay in them
        #also only social people do social movement
        #no check to stay at home, since percentage of social will be reduced to essential workers
        if (daily_hour <= 8 and self.social):
            pos_X = self.location[0]
            pos_Y = self.location[1]
            
            
            #if we are at a social area, move within it
            if (grids[self.location[0], self.location[1]] and self.social):
                
                #decide how many grid units he will move within social areas (1 to 5)
                travel = np.random.randint(1,6)
                
                #obtain possible indices in social areas
                masked = np.zeros((max_size,max_size))
                masked[pos_X-travel:pos_X+(travel+1)  ,pos_Y-travel:pos_Y+(travel+1)] = grids[pos_X-travel:pos_X+(travel+1)  ,pos_Y-travel:pos_Y+(travel+1)]
                possx, possy = np.nonzero(masked)
                
                
                #select random place in social area to move to and do so
                rand_pos = np.random.randint(len(possx))
                
                npos_X = possx[rand_pos]
                npos_Y = possy[rand_pos]
                
                self.location = (npos_X, npos_Y)
                return (npos_X,npos_Y)
                
            #we are not at a social area
            #if not, look for closest area and go towards it
            else:
            
                
                #calculate distances to centroids and locate closest centroid
                distances = distance.cdist(centroids, np.array([self.location]))
                closest_c = centroids[distances.argmin()]
                
                
                #obtain distance to travel
                d_travel_hour= abs(np.random.normal(40,5))
                move_increment_grid= d_travel_hour/2 #since each grid is 2x2 meters
                
                #see if distance to travel is bigger than distance to centroid, to avoid overshooting 
    
                #distance is greater, so new position will be centroid + slight change to avoid people concentrating
                if (move_increment_grid > distances[distances.argmin()]):
                    #setting centroid as destination and adding slight change
                    npos_X = closest_c[0] + np.random.randint(-5,6)
                    npos_Y = closest_c[1] + np.random.randint(-5,6)
                    
                #otherwise go to predetermined place    
                else:
                    #calculate direction vector
                    direc_vec = np.subtract(closest_c, self.location)
                    
                    #calculate new positions using distance and movement (casting to int)
                    newpos = self.location + ((move_increment_grid / distances.min())* direc_vec)
                    
                    npos_X= int(newpos[0])
                    npos_Y= int(newpos[1])
                    
                
                self.location = (npos_X, npos_Y)
                return (npos_X,npos_Y)
                
            
        
        
        
        
        #rest of the day, people will move randomly 
        else:
            
            #in case of quarantine, people who are social and % of population will not do quarantine
            # these represent esential workers who move thru the map and % of people who will still move
            
            
            
            #this if represents cases when there is no stay at home order (normal movement), or when there
            #is that order, only social and % of people who dont follow it move
            # hese represent esential workers who move thru the map and % of people who will still move
            if ( not(stay_home) or ( stay_home and ((self.freeroam) or (self.social)))):
                
        
                #draw how many meters the agent will move in the timestep
                #mean is fixed at 50 meters (references)
                d_travel_hour = np.random.normal(25,5)
                if (d_travel_hour < 0): #done to avoid rare chances where 
                    d_travel_hour = 0
                move_increment_grid= d_travel_hour/2 #since each grid is 2x2 meters 
                
                pos_X= self.location[0]
                pos_Y= self.location[1]
                
                direction= np.random.uniform(0,2*pi) # random unit circle search radius for direction
                
                npos_X= int(pos_X+ move_increment_grid*cos(direction))
                npos_Y= int(pos_Y+ move_increment_grid*sin(direction))
                
               
                #if we are out of bounds, pick new direction, this method is faster than checking every single possibility
                while (npos_X<0 or npos_Y<0 or npos_X>=max_size or npos_Y>=max_size):
                    
                    direction=direction+(pi/8)
        
                    npos_X= int(pos_X+move_increment_grid*cos(direction))
                    npos_Y= int(pos_Y+move_increment_grid*sin(direction))
                    
                
                
                #panic factor will determine social isolation (only if its higher than 1)
                #no need to look for place with less density if i am alone, people in freeroam dont do social distancing
                if (panic_factor >= 1 and self.freeroam == 0 and pop_density[npos_X,npos_Y] != 0):
                
                    #used to calculate distance in grid units    
                    radius = panic_factor #this is for now
                    
                    #create square of indices
                    X = np.arange(npos_X-radius, npos_X+radius+1)
                    Y = np.arange(npos_Y-radius, npos_Y+radius+1)
                    
                    #delete the indices that are out of bounds
                    X = X[(X>=0) & (X<max_size)]
                    Y = Y[(Y>=0) & (Y<max_size)]
                    
                    xposall, yposall = np.meshgrid(X,Y)
                    
                    #create density array where all densities will be saved to later find the minimum
                    densities = np.zeros(len(X)*len(Y))
                    
                    ix = 0
                    for x,y in zip (xposall.flatten(), yposall.flatten()):
                        densities[ix] = pop_density[x,y] 
                        ix = ix+1
                
                    
                     #obtain min index, if there is more than 1, a random one will be chosen
                    min_idxs = np.where(densities == densities.min())
                    
                    idx_min_dist = np.random.randint(len(min_idxs[0]))
                    
                    #obtain the actual index
                    npos_X = xposall.flatten()[min_idxs[0][idx_min_dist]]
                    npos_Y = yposall.flatten()[min_idxs[0][idx_min_dist]]
                    
                    
                    
                self.location = (npos_X,npos_Y )
                return (npos_X,npos_Y)
            
            #if this else is reached, this means that we are in stay at home order, and
            #these people will only move with a small set percentage, in order to represent people 
            # who sometimes move to buy groceries, walk their dogs, do exercise, etc
            else:
                
                pos_X= self.location[0]
                pos_Y= self.location[1]
                
                
                #chance of moving in stay at home order for "normal" people
                if ( stay_move_chance > np.random.uniform(0, 1)):
                    direction= np.random.uniform(0,2*pi) # random unit circle search radius for direction
                    
                    d_travel_hour= np.random.normal(25,5)
                    if (d_travel_hour < 0): #done to avoid rare chances where 
                        d_travel_hour = 0
                    move_increment_grid= d_travel_hour/2 #since each grid is 2x2 meters
                
                    npos_X= int(pos_X+ move_increment_grid*cos(direction))
                    npos_Y= int(pos_Y+ move_increment_grid*sin(direction))
                    
                    #if we are out of bounds, pick new direction, this method is faster than checking every single possibility
                    while (npos_X<0 or npos_Y<0 or npos_X>=max_size or npos_Y>=max_size):
                    
                        direction=direction+(pi/8)
            
                        npos_X= int(pos_X+move_increment_grid*cos(direction))
                        npos_Y= int(pos_Y+move_increment_grid*sin(direction))
                    
                #we stay where we are    
                else:
                    npos_X = pos_X
                    npos_Y = pos_Y
                
                    
                    
            
                self.location = (npos_X,npos_Y )
                return (npos_X,npos_Y)







    def survived_dt (self):
        """
        action that will be called per infected individual if he hasnt died beforehand
        will increase t_infected and check if it is equal to time to recuperate
        in case that they are equal, the agent will be updated to inmune and the method will return
        the corresponding information
        

        Returns
        -------
        status:
            0 if agent still has dts until healthy again
            1 if agent has survived all timesteps

        """
        status = 0
        
        self.t_infected += 1
        if (self.t_infected == self.t_torecuperate):
            status = 1
            self.health = 2
            
        return status
        



    ########################
    #GETTERS
    ########################
        
    def get_age(self):
        return self.age
    
    def get_location(self):
        return self.location
    
    def get_x(self):
        return self.location[0]
    
    def get_y(self):
        return self.location[1]
    
    def get_t_infected(self):
        return self.t_infected
    
    def get_health(self):
        return self.health
    
    def get_quarantine(self):
        return self.quarantine
    
    def get_immunocompromised(self):
        return self.immunocompromised
    
    def get_t_torecuperate(self):
        return self.t_torecuperate
    
    def get_chance_death_dt (self):
        return self.chance_death_dt
    
    def get_freeroam (self):
        return self.freeroam
    
    def get_social (self):
        return self.social
    
    
    
    
    ########################
    #SETTERS
    ########################
    
    def set_age(self, age):
         self.age = age
    
    def set_location(self, location):
         self.location = location
    
    def set_t_infected(self, t_infected):
         self.t_infected = t_infected
    
    def set_health(self, health):
         self.health = health
    
    def set_quarantine(self, quarantine):
         self.quarantine = quarantine
    
    def set_immunocompromised(self, immunocompromised):
         self.immunocompromised = immunocompromised 
         
    def set_t_torecuperate(self, t_torecuperate):
        self.t_torecuperate = t_torecuperate
        
    def set_chance_death_dt (self, chance_death_dt):
        self.chance_death_dt = chance_death_dt
        
    def set_freeroam (self, freeroam):
        self.freeroam = freeroam
        
    def set_social (self, social):
        self.social = social
        
        
        
        
    
        
        