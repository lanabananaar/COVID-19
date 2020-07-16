#!/usr/bin/python

import os
import pickle
import sys
import argparse
from argparse import RawTextHelpFormatter
from matplotlib import pyplot as plt
import human as agent_class
import numpy as np
import numpy.matlib
import pandas as pd
from math import ceil

PARAMS = dict() #will contain parameters used
ENV_DATA = dict() #will contain environment data
IT_STATS = dict() #will contain results per day and dt


def progressbar(acc, total, total_bar, dt, day):
    """
    auxiliary function used to print in console % of experiment done and
    current dt and day. Used to estimate completion

    Parameters
    ----------
    acc : int
        current timestep (accumulated).
    total : int
        total amount of timesteps.
    total_bar : int
        width of the bar (50 is a good value).
    dt : int 
    day : int
    """
    
    
    frac = acc/total
    filled_progbar = round(frac*total_bar)
    print('\r', '#'*filled_progbar + '-'*(total_bar-filled_progbar) + ' dt: ', str(dt) + ' day: ', str(day), '[{:>7.2%}]'.format(frac), end='')





def death_per_dt_aux(dts, age, immuno):
    """
    Based on age and dts until recuperation, calculates chances
    of dying in each timestep

    Parameters
    ----------
    dts : int
        age of agent
    dts : int
        number of timesteps to pass infection.

    Returns
    -------
    chance of death per timestep.

    """
    immuno_mult = 1
    
    if (immuno):
        immuno_mult = 1.089
    
    total_chance_alive =   1 - (PARAMS["death_probabilities_age"][age] * immuno_mult) #chance of surviving the infection
    return ( -(total_chance_alive**(1/dts)) + 1)  
    
    pass



def create_params(rnaught, duration_infection, contacts_day, nag, sick_time, day_measure, social_distancing, stay_at_home, mov_chance_stay_home, seed):
    """
    Sets up the corresponding parameters for the simulation. 
    The structure in question is PARAMS, where behaviour will arise from the
    interplay of said PARAMS. PARAMS is a dictionary

    Parameters
    ----------
    rnaught : int
        r0 of COVID-19
    
    duration_infection : int
        mean of average time to recuperate from sickness (in dts)
    
    contacts_day : int
        mean of number of contacts (close contacts) for a human per day
    
    nag : number of agents
        total number of agents that the simulation started with
        
    day_measure: int
        day when measures are implemented
        
    social_distancing: int
        social distancing square radius
        
    stay_at_home: int
        stay at home order status (1 if activated, 0 otherwise)
        
    mov_chance_stay_home: float
        percentage of moving per timestep of agents who stay at home
        
    seed : int
        seed of the simulation
    
    
    """
    
    #calculating % of infection on contact (in grid) based on r0, average
    #infection duration of the virus, and number of contacts per day
    #if r0 is average number of people infected by a single infected individual
    #we can make the following assumptions based on number of contacts per day
    #and average length of virus infection
    
    infect_prob = rnaught / ((duration_infection/16) * contacts_day)
    
    PARAMS["r_0"] = rnaught
    PARAMS["chance_infect"] = infect_prob #chance of infection in same  (from 1 to 0)
    PARAMS["dt"] = 0 # number of timesteps performed
    PARAMS["initial_agents"] = nag
    PARAMS["sick_duration_mean"] = sick_time
    PARAMS["day_that_measures_are_implemented"] = day_measure
    PARAMS["social_distancing_radius"] = social_distancing
    PARAMS["stay_at_home_order"] = stay_at_home
    PARAMS["mov_chance_stay_home"] = mov_chance_stay_home
    PARAMS["seed"] = seed
    
    
    #creating the array where the index == age and value == % of dying if infected with covid-19
    imported_data = pd.read_excel('deathsbyagegroup.xlsx', usecols = ['Range','Death_prob'])
    age_death_dists = imported_data['Range'].tolist()
    death_chances = imported_data['Death_prob'].tolist()
    
    death_chance_array = []
    
    for minmaxstring,percentage in zip(age_death_dists, death_chances):
        
        minmaxage = minmaxstring.split('-')
        
        for _ in range (int(minmaxage[0]), int(minmaxage[1])+1):
            death_chance_array.append(percentage)
        
    
    PARAMS["death_probabilities_age"] = death_chance_array
    
    






    
def create_environment(size):
    """
    creates the global data structure for the environment
    the data structure is a matrix of 2x2 meter cells
    These cells are either normal or highly social (workplaces, universities, etc) 
    "Normal cells"        : Represented by a 0
    "Highly Social cells" : Represented by a 1


    The environment is fixed, since it is a 1/9 scale model of sheffield
    It is a "square" version of the inner urban city area, where highly
    social areas primarily workplaces and universities, and certain office
    buildings and other areas

    size has been calculated as 75, it is left as an argument in case it is
    later changed, and for code readibility
    
    a 1/9 model would have been 150x150, but it was resized to 75 because the simulations
    were far too slow and because of possible memory limitations
    

    Parameters
    ----------
    size : int
        number of 2x2 meter cells by side



    """
    
    # Following the distribution of sheffield, we will roughly create areas
    # in order to create big social places like the
    # ones illustrated in the original map

    #sizes of the predetermined areas that summed up make 21% of the total area
    area1 = np.ones((37,20))
    area2 = np.ones((10,10))
    area3 = np.ones((15,15))
    area4 = np.ones((10,10))
    
    
    #TOTAL NUMBER OF HIGHLY SOCIALLY ACTIVE GRIDS: 1165
    #TOTAL NUMBER OF GRIDS: 5625
    #PERCENTAGE OF HIGHLY SOCIAL GRIDS: 20.7% (PRE-CALCULATED)
    
    ENV_DATA["size"] = size
    ENV_DATA["units"] = "meters"
    ENV_DATA["grids"] = np.zeros((size,size))
    ENV_DATA["grids"][24:61 , 41:61] = area1
    ENV_DATA["grids"][29:39 , 5:15] = area2
    ENV_DATA["grids"][34:49 , 19:34] = area3
    ENV_DATA["grids"][54:64 , 26:36] = area4
    
    #centroids of the grid areas used for social movement
    ENV_DATA["centroids"] = np.array([[42,51],[34,10],[41,26],[59,30]])
    
    #These are used for calculations and determining where infections can happen    
    ENV_DATA["population_density"] = np.zeros((size, size)) #number of agents per grid
    ENV_DATA["infected_density"] = np.zeros((size, size)) #number of infected agents per grid
    ENV_DATA["susceptible_density"] = np.zeros((size, size)) #number of "healthy" individuals per grid
    
    #We will use flat indexes and tuple of index arrays to our advantage (to avoid looping
    #through all human agents to find x and y positions). We will create a list of lists, where the length
    #will be the total amount of grids. This way, we will use flat indices to store our human agents and convert 
    #to x,y positions when needed and viceversa
    
    ENV_DATA["agents_flat_positions"] = [[] for _ in range(size*size) ]




def initialise_results(dts, o_nag, o_infect, o_immuno, o_freeroam, o_social):
    """
    Creates the dictionary (data structure) that will record important statistics
    for both timesteps and daily occurrences (every 16 timesteps)

        Parameters
        ----------
        dts : int
            maximum number of timesteps in the simulation
        o_nag : int
            original number of (alive) individuals in the simulation
        o_infect : int
            original number of infected individuals in the simulation
        o_immuno : int
            original number of immunocompromised individuals in the simulation

    """
    #BASIC INFORMATION
    IT_STATS["total_initial_agents"] = o_nag
    IT_STATS["total_initial_infected"] = o_infect
    IT_STATS["total_initial_immuno"] = o_immuno
    IT_STATS["total_free_roam_agents"] = o_freeroam
    IT_STATS["total_social_agents"] = o_social
    IT_STATS["current_dt"] = 1
    IT_STATS["dt_of_the_day"] = 1
    IT_STATS["current_day"] = 1
    
    
    #INFORMATION PER TIMESTEP OR DAY
    IT_STATS["total_alive_per_timestep"] = np.zeros(dts)
    IT_STATS["total_alive_per_timestep"][0] = o_nag
    IT_STATS["total_alive_per_day"] = np.zeros(ceil(dts/16))
    
    IT_STATS["total_infected_per_timestep"] = np.zeros(dts)
    IT_STATS["total_infected_per_timestep"][0] = o_infect
    IT_STATS["total_infected_per_day"] = np.zeros(ceil(dts/16))
    IT_STATS["total_infected_per_day"][0] = o_infect

    
    IT_STATS["total_immune_per_timestep"] = np.zeros(dts)
    IT_STATS["total_immune_per_day"] = np.zeros(ceil(dts/16))
    
    IT_STATS["total_quarantined_per_timestep"] = np.zeros(dts)
    IT_STATS["total_quarantined_per_day"] = np.zeros(ceil(dts/16))
    
    IT_STATS["total_susceptible_per_timestep"] = np.zeros(dts)
    IT_STATS["total_susceptible_per_timestep"][0] = o_nag - o_infect
    IT_STATS["total_susceptible_per_day"] = np.zeros(ceil(dts/16))
    IT_STATS["total_susceptible_per_day"][0] = o_nag - o_infect

    
    
    
    
    
    
    IT_STATS["deaths_per_timestep"] = np.zeros(dts)
    IT_STATS["deaths_per_day"] = np.zeros(ceil(dts/16))
    
    IT_STATS["infections_per_timestep"] = np.zeros(dts)
    IT_STATS["infections_per_day"] = np.zeros(ceil(dts/16))
    
    IT_STATS["recovered_per_timestep"] = np.zeros(dts)
    IT_STATS["recovered_per_day"] = np.zeros(ceil(dts/16))
    
    IT_STATS["quarantined_per_timestep"] = np.zeros(dts)
    IT_STATS["quarantined_per_day"] = np.zeros(ceil(dts/16))
    







def create_agents(nagi, age_all, nr_infec, nr_immucmpr, nr_freeroam, nr_social):
    """
    Creates the array of agents that will be used in the simulation

    Parameters
    ----------
    nagi : int
        number of human agents that will be created.
    age_all : numpy array of ints
        distribution of the ages of the agents
    nr_infec : int
        number of initial infected people (chosen randomly)
    nr_immucmpr : int
        number of immunocompromised people (chosen randomly)
    nr_social: int
        number of social people who will go to social areas in first 8 hours of the day
    sick_duration : int
        number of hours of average infection duration

    Returns
    -------
    humans
    list that contains all the human objects created

    """
    humans = []

    
    max_size = ENV_DATA["size"]
    nr_of_grids = max_size * max_size #used for linear indices
    
    #create random positions (linear indexing) for the agents
    pos_all = np.random.randint(nr_of_grids, size=nagi)
    
    
    
    #create random indices (without repetition) of infected and immunocompromised individuals, also of freeroam and social agents
    initial_infected_pos = np.random.choice(nagi, nr_infec, replace=False)
    immunocomp_pos = np.random.choice(nagi, nr_immucmpr, replace=False)
    freeroam_pos = np.random.choice(nagi, nr_freeroam, replace = False)
    social_pos = np.random.choice(nagi, nr_social, replace = False)
    
    #create random distribution of timesteps until recuperation (only for infected)
    dts_until_healthy = (np.random.normal(PARAMS["sick_duration_mean"],48,nr_infec)).astype(int)
    dts_until_healthy = dts_until_healthy.clip(min=0) #avoid incredibly rare chances where value is negative
    
    #fill positions of immuno, freeroam and social
    immnuno_all = np.zeros(nagi)
    immnuno_all[immunocomp_pos] = 1
    
    freeroam_all = np.zeros(nagi)
    freeroam_all[freeroam_pos] = 1

    social_all = np.zeros(nagi)
    social_all[social_pos] = 1

    
    #create chance of death per timestep (only for infected)
    death_per_dt_chances = map(death_per_dt_aux, dts_until_healthy, age_all[initial_infected_pos], immnuno_all )
    
    #create our health, immunocompromised, time until healthy and dt chance death arrays (see human.py for futher reference)
    health_all = np.zeros(nagi)
    time_to_healthy_all = np.negative(np.ones(nagi))
    chance_death_dt_all = np.negative(np.ones(nagi))
    
    
    #here 1 means that they are infected (health) and that they are immunocompromised (immuno)
    health_all[initial_infected_pos] = 1
    time_to_healthy_all[initial_infected_pos] = dts_until_healthy
    chance_death_dt_all[initial_infected_pos] = list(death_per_dt_chances)

    
    
    
    
    #creation of the array of agents
    for i in range(nagi):
        age = age_all[i]
        pos = tuple(np.unravel_index(pos_all[i],(max_size,max_size)))
        health = health_all[i]
        immunocompromised = immnuno_all[i]
        time_until_healthy = time_to_healthy_all[i]
        chance_death_dt = chance_death_dt_all[i]
        freeroam = freeroam_all[i]
        social = social_all[i]
        human = agent_class.Human(age, pos, 0, health, 0, immunocompromised, time_until_healthy, chance_death_dt, freeroam, social)
        humans.append(human)
        if (health == 1):
            ENV_DATA["infected_density"][pos[0],pos[1]] += 1  #density of infected individuals
        else:
            ENV_DATA["susceptible_density"][pos[0],pos[1]] += 1 #addint 1 to selected susceptible density (used for infect method)
            
        ENV_DATA["population_density"][pos[0],pos[1]] += 1  #adding 1 to selected density grid area
        
        #flat indexing of the human objects
        (ENV_DATA["agents_flat_positions"][np.ravel_multi_index((pos[0],pos[1]), (max_size,max_size))]).append(human)



    return humans





def age_dist (n_agents):
    """
    Creates array of ages to be used in agent creation for their age
    The data comes from the Office of National Statistics, UK,
    Population estimation of mid 2018
    
    This methods calculates the probability of the an agent belonging to the    
    the different age options via its CDF and sampling from this distribution
    
    NOTE: Last entry (row 92 of Age column) was originally "90+", the +
    was eliminated manually, and a maximum age of 90 was set for this simulation,
    this will not have an impact since ages of 90 or more would be considered as the
    same for infection/death purposes

    Parameters
    ----------
    n_agents : int
    number of ages to be calculated (number of agents in simulation)

    Returns
    -------
    ages : numpy array of ints
        array with all the ages of the individuals
        ages range from 0 (babies) to 90 (maximum age of elderly people)

    """
    
    #importing only the useful data (age and demographic of that age)
    imported_data = pd.read_excel('population_estimate_2018_ONS.xlsx', usecols = ['Age','All'])
    
    #get all possible values in a list and eliminate the last row which we dont need
    #possible_ages = (imported_data['Age'].tolist())[:-1] #used to check with original data if neccesary
    age_demographic = np.array((imported_data['All'].tolist())[:-1])
    total_pop = imported_data['All'][len(imported_data['All'])-1] #total population is located here
    
    
    y_perc = age_demographic / total_pop #percentage of individuals of each age group 
    y_sum_cdf = np.cumsum(y_perc) # CDF of age group
    
    
    x_random = np.random.uniform(size = n_agents) # drawing random distributed numbers
    
    #here what we are doing is substracting the random uniform numbers from our CDF 
    #when we do the absolute, we ara left with arrays where the minimum value would be
    #our sample. We use argmin to obtain the position of said min value and we have our age
    ages = np.argmin(abs(x_random.reshape(n_agents, 1) - y_sum_cdf ), axis = 1 )

    return ages
    

    
def set_random (useseed, seed, verbose):
    """
    

    Parameters
    ----------
    useseed : logical int (1 or 0)
        If 1 = user wants specified random seed, 0 otherwise
    seed : int
        Random seed to be set



    """
    
    if (useseed):
        if (seed == -1):
            sys.exit("usage of selected seed was desired, but no seed was specified, please use -h command and call the program correctly")
        np.random.seed(seed)
        
        if (verbose):
            print("The random seed has been set to: {}".format(seed))
            print("As such, the same random values, wherever drawed, will be the same if the same seed is used ")

    
    
    
    




def infect (agents, dt, day):
    """
    This method checks whether or not an infection is possible (at least 2 or more people in
    a grid, while one is infected and the other is susceptible)
    
    The operations are done in such a way that randomness is not needed, since infections are calculated per grid.
    There is no possible advantage in looping through the environment in any way, nor is it important if one human gets infected
    later or sooner, due to it all being in a single timestep
    
    Parameters
    ----------
    humans : list of human objects
        Contains all the agents in the simulation
    dt: int
        Current dt
    day: int
        Current day
  
    """
    
    #obtain the parts of the grid where there is a possibility of infection
    #that is, at least 1 susceptible individual and at least 1 infected individual
    #we can check this by multiplying both population matrixes element wise and
    #obtaining all non-zero values (agents are stored in a flat-like list)
    
    infection_chance_flats = np.flatnonzero(ENV_DATA["susceptible_density"] * ENV_DATA["infected_density"])
    np.random.shuffle(infection_chance_flats)
    
    infected_total = 0 #number of infected agents, used to update IT_STATS
    
    mean_sick_time = PARAMS["sick_duration_mean"]
    
    #x_y_pairs = [ np.unravel_index(x, (ENV_DATA["size"],ENV_DATA["size"])) for x in infection_chance_flats ]
    
    for i in infection_chance_flats:
        agents_grid = ENV_DATA["agents_flat_positions"][i] #we do this here because we need to access this twice for the list comprehension
        xypos = np.unravel_index(i, (ENV_DATA["size"],ENV_DATA["size"])) #used to update infected and susceptible densities
        
        #we get agents that may be infected, that is, susceptible agents
        #if health == 0, not 0 will be a logical 1, therefore true.
        
        susceptible_humans = [healthy for healthy in agents_grid if not(healthy.get_health()) ]
        nr_infec_grid = len([infected for infected in agents_grid if (infected.get_health() == 1)])
        
        #we draw x(number of susceptible) y(number of infected) times
        #each row is each agent, so we will later check if any random prob in the row is < infection chance
        random_chances = np.random.uniform(size = (len(susceptible_humans),nr_infec_grid ))
        #find indices if there is a value in each row that is true from np.any (which checks if there was such a value per row)
        to_debug = np.any((PARAMS["chance_infect"] > random_chances), axis=1)
        infect_idxs = np.argwhere(to_debug).flatten() #flattened for it to be iterable in for loop
        
        #calculation of time until each agent gets healthy
        dts_until_healthy = (np.random.normal(mean_sick_time,48,len(infect_idxs))).astype(int)
        
        

        #we change the health of susceptible agents to infected
        for e,j in enumerate(infect_idxs):
            susceptible_humans[j].set_health(1)
            susceptible_humans[j].set_t_torecuperate(dts_until_healthy[e])
            
            death_chance_per_dt = death_per_dt_aux(dts_until_healthy[e], susceptible_humans[j].get_age(), susceptible_humans[j].get_immunocompromised())
            susceptible_humans[j].set_chance_death_dt(death_chance_per_dt)
            
            
            #we update our corresponding grids
            ENV_DATA["infected_density"][xypos] += 1
            ENV_DATA["susceptible_density"][xypos] -= 1
            
        infected_total += infect_idxs.size
            
    
        
          
    IT_STATS["infections_per_timestep"][dt] = infected_total
    IT_STATS["infections_per_day"][day] += infected_total
        
    IT_STATS["total_susceptible_per_timestep"][dt] =  np.sum(ENV_DATA["susceptible_density"])
    IT_STATS["total_infected_per_timestep"][dt] =  np.sum(ENV_DATA["infected_density"])
    
    
    
    if ((IT_STATS["dt_of_the_day"] % 16) == 0):
        IT_STATS["total_infected_per_day"][day] =  np.sum(ENV_DATA["infected_density"])
        IT_STATS["total_susceptible_per_day"][day] =   np.sum(ENV_DATA["susceptible_density"])

        
        
        
        
      
        




def death(agents, quarantined_agents, dt, day):
    """
    Goes over infected agents and essentially updates the number of dts the agent
    has been infected for, running the probability of dying and updating the corresponding
    data structures

    Parameters
    ----------
    agents : list of agent objects
        list of active agents.
    quarantined_agents : lit of agent objects 
        lit of active objects who are quarantined.
    dt : int
        current dt.
    day : int
        current day.

    """
    
    #see where there are infected individuals
    places_to_check = (np.flatnonzero(ENV_DATA["infected_density"]))
    np.random.shuffle(places_to_check)
                    
    nr_deaths = 0; #will record how many die
    nr_immune = 0;#how many will pass the infection
    
    for i in places_to_check:
        #get the infected agents
        infected = [j for j in ENV_DATA["agents_flat_positions"][i] if j.get_health() == 1] 
        
        #draw random chances for death for each of the infected agents
        random_chances = np.random.uniform(size = len(infected))

        for agent, chance in zip(infected, random_chances):
            
            #check if agent has died
            if (agent.get_chance_death_dt() > chance  ):
                #agent has died
                
                agent.set_health(3)
                nr_deaths += 1
                
                #update ENV_DATA and others
                xypos =  agent.get_location()
                ENV_DATA["population_density"][xypos] -= 1
                ENV_DATA["infected_density"][xypos] -= 1
                ENV_DATA["agents_flat_positions"][i].remove(agent)
                agents.remove(agent)
                
            #infected agent survived one timestep    
            else:
                
                if (agent.survived_dt()):
                    xypos =  agent.get_location()
                    ENV_DATA["infected_density"][xypos] -= 1
                    nr_immune += 1
            
                    
    #we now need to check if our quarantined agents die or not
    death_chances = [t.get_chance_death_dt() for t in quarantined_agents]
    random_chances = np.random.uniform(size = len(death_chances))   
         
    for ix, q_agent in enumerate(quarantined_agents):
        if (death_chances[ix] > random_chances[ix]):
            #agent has died
            q_agent.set_health(3)
            nr_deaths += 1
                
            #update
            quarantined_agents.remove(q_agent)
            ENV_DATA["population_density"][q_agent.get_location()] -= 1
            
        else:
            #if quarantined agent survives, he is not susceptible, but is added back into normal population
            #but he is not added into susceptible density, only population density
            if (q_agent.survived_dt()):
                
                nr_immune += 1
                quarantined_agents.remove(q_agent) #no longer quarantined
                agents.append(q_agent) #added to active agents
                
                #is now part of population and added to flat indices
                flatpos = np.ravel_multi_index(q_agent.get_location(),(ENV_DATA["size"],ENV_DATA["size"]))
                
                ENV_DATA["population_density"][q_agent.get_location()] += 1
                (ENV_DATA["agents_flat_positions"][flatpos]).append(q_agent)

        
        
        

    #update corresponding death statistics
    IT_STATS["deaths_per_timestep"][dt] = nr_deaths
    IT_STATS["deaths_per_day"][day] += nr_deaths
    
    
    #update corresponding recovered statistics
    IT_STATS["recovered_per_timestep"][dt] = nr_immune
    IT_STATS["recovered_per_day"][day] += nr_immune
        







def quarantine(agents, quarantined_agents, dt, day):
   """
    quarantines infected agents based on different procedures. Currently not used in simulations
    but could be called and works. Careful with detemining the policy for quarantining

    Parameters
    ----------
    agents : list
        active agents in the simulation.
    quarantined_agents : list
        quarantined agents in the simulation.

    """
    #randomise in order to avoid preferences
   places_to_check = np.flatnonzero(ENV_DATA["infected_density"])
   np.random.shuffle(places_to_check)
   nr_quarantined = 0 #people quarantined
   
   
    
    
   for i in places_to_check:
        #get the infected agents
        infected = [j for j in ENV_DATA["agents_flat_positions"][i] if j.get_health() == 1] 
        
        #This determines the policy for quarantining
        random_chances = np.random.randint(106, 124, size = len(infected)) #get dts to quarantine
        
        for agent, chance in zip(infected, random_chances):
            
            if (agent.get_t_infected() > chance):
                
                agent.set_quarantine(1)
                nr_quarantined += 1
                
                #update ENV_DATA and others
                xypos =  agent.get_location()
                ENV_DATA["population_density"][xypos] -= 1
                ENV_DATA["infected_density"][xypos] -= 1
                ENV_DATA["agents_flat_positions"][i].remove(agent)
                agents.remove(agent)
                quarantined_agents.append(agent)
                
        

   IT_STATS["quarantined_per_timestep"][dt] = nr_quarantined
   IT_STATS["quarantined_per_day"][day] += nr_quarantined
   
   if ((IT_STATS["dt_of_the_day"] % 16) == 0):
       
        IT_STATS["total_quarantined_per_day"][day] =  len(quarantined_agents)







def update(day, day_to_implement, new_social_distance, new_stay_at_home):
    """
    updates current policies, panic, quarantine policiy and stay at home order
    Parameters
    ----------
    day : int
        current day of the simulation.
    day_to_implement : int
        day from which measures are implemented.
    new_social_distance : int
        radius of square in which agent will look for other less populated areas.
    new_stay_at_home : int
        determines if stay at home order will be activated.

    Returns
    -------
    updated panic index, quarantine policy and stay at home order

    """    
    panic = 0
    quarantine_policy = 0
    stay_home = 0
    
    if (day >= day_to_implement-1):
        panic = new_social_distance
        quarantine_policy = 0
        stay_home = new_stay_at_home
        
        
    
    return panic, quarantine_policy, stay_home



def move (agents, panic_factor, i, day, stay_home, stay_move_chance):
    """
    calls move function for each different agent, updating the corresponding
    data structures 

    Parameters
    ----------
    agents : list with human objects
        contains all the active agents in the simulation.
    i : int
        contains actual timestep of the simulation.
    day : int
        contains current day of the simulation.
        
    panic_factor: int
        current panic factor
        
    stay_home: int
        current status of stay at home order
        
    stay_move_chance: float
        percentage of moving per timestep of agents who stay at home

    """
    #randomize order or agents to avoid possible advantages
    
    np.random.shuffle(agents)
    

    max_size = ENV_DATA["size"]
    
    for ix, agent in enumerate(agents):
        oldxypos = agent.get_location()
        newxypos = agent.movev2(max_size, IT_STATS["dt_of_the_day"], ENV_DATA["population_density"],
                                panic_factor, ENV_DATA["centroids"], ENV_DATA["grids"], stay_home, stay_move_chance)
        
        #update population density
        ENV_DATA["population_density"][oldxypos] -= 1
        ENV_DATA["population_density"][newxypos] += 1
        
        #update flat_positions
        (ENV_DATA["agents_flat_positions"][np.ravel_multi_index(oldxypos,(max_size,max_size))]).remove(agent)
        (ENV_DATA["agents_flat_positions"][np.ravel_multi_index(newxypos,(max_size,max_size))]).append(agent)

        #now we have to update the corresponding infected or susceptible
        
        if(agent.get_health()): #if health == 1 agent is infected
            ENV_DATA["infected_density"][oldxypos] -= 1
            ENV_DATA["infected_density"][newxypos] += 1
        else:
            ENV_DATA["susceptible_density"][oldxypos] -= 1
            ENV_DATA["susceptible_density"][newxypos] += 1
            
        

def main(raw_args=None):
     
    
    pars = argparse.ArgumentParser(description = "ABM host-pathogen SIR model which simulates a scaled down version of the \
    city of Sheffield, where infetec and healthy agents interact on a daily basis. \n COVID-19 Simulator. Group Assignment for COM6009.\n \
    AUTHORS: Jorge Díez, Lana Abou, Jingze Liu \n University of Sheffield, 2020 \n \
    To give values for arguments, please type into the call -argument VALUE, for example:\n If we wanted to set the \
    number of agents to 10000 and the maximum number of timesteps to 40000 and leave \
    the other options as default, you would need to write the following: \n python covid.py -nag 10000 -ndts 40000", formatter_class = RawTextHelpFormatter)
    
    pars.add_argument('-nag', type=int, default=5000, help='Number of agents')
    pars.add_argument('-r0', type=float, default=2.5, help='R0 of COVID-10')
    pars.add_argument('-ndts', type=int, default=4320, help='Maximum number of timesteps. 1 timestep = 1 hour. 16 dts = 1 day')
    pars.add_argument('-ninfect', type=int, default=3, help='Initial number of infected agents')
    pars.add_argument('-perc_immuno', type=float, default=0.178, help='Percentage of immunocompromised agents (including people with udnerlying health conditions')
    pars.add_argument('-perc_social', type=float, default = 0.6, help='Percentage of move who move to social areas in first 8 hours of the day')
    pars.add_argument('-perc_free', type=float, default=0.4, help='Percentage of people who dont isolate, quarantine or use social panic')
    pars.add_argument('-sick_time', type=int, default=224, help='Mean of average length agent remains infected in timesteps (hours)')
    pars.add_argument('-contact_day', type=int, default=20, help='Average number of contacts between people, used to obtain chance of infection')
    pars.add_argument('-random', type=int, default=0, help='1 to get same results as other simulations 0 otherwise.')
    pars.add_argument('-seed', type=int, default=-1, help='if random was activated, this seed number will give back same results as other experiments with same seed')
    pars.add_argument('-foldersave', type=str, default = 'No folder input', help='if set, results will be automatically saved and no figures outputted')
    pars.add_argument('-v', type=int, default = 0, help='if 1, verbose outputted. default is 0')
    pars.add_argument('-visualize', type=int, default = 0, help='if 1, will show how the population moves per timestep')
    pars.add_argument('-day_measure', type=int, default = 0, help='Day from which measures are taken')
    pars.add_argument('-social_distancing', type=int, default = 0, help='Determines square size in grids that agent will look for to avoid concentrations')
    pars.add_argument('-stay_at_home', type=int, default = 0, help='Determines if government stay at home order will be put into place')    
    pars.add_argument('-move_every_x_days', type=float, default = 2.5, help='How often do stay at home people move once measured in days ')    
    pars.add_argument('-visualize_infected', type=int, default = 0, help='if 1, will show total infected per timestep , computationally demanding')
    args=pars.parse_args(raw_args)
    
    if (args.v):
        
        
        print("\n###########################################################\n")
        print ("The following parameters have been set for the simulation: ") 
        print("Number of agents: {}".format(args.nag))
        print("R0: {}".format(args.r0))
        print("Max dts: {}".format(args.ndts))
        print("Initial infected: {}".format(args.ninfect))
        print("Percentage of immunocompromised : {}".format(args.perc_immuno))
        print("Percentage of social : {}".format(args.perc_social))
        print("Percentage of free_roam : {}".format(args.perc_free))
        print("Average infection duration: {}".format(args.sick_time))
        print("Random simulation selector: {}".format(args.random))
        print("Seed selected for random: {}".format(args.seed))
        print("Folder where results will be saved : {}".format(args.foldersave))
        print("\n###########################################################\n")
        
    
    
    nimmuno = int(args.perc_immuno * args.nag)
    nfree = int(args.perc_free * args.nag)
    nsocial = int(args.perc_social * args.nag)
    
    #calculate % of moving in each timestep based on how often stay at home agents move 
    #once every x days
    
    mov_chance_stay_home = 1/(args.move_every_x_days * 16)
    
    
    create_params(args.r0, args.sick_time, args.contact_day,args.nag, args.sick_time, args.day_measure,
                  args.social_distancing, args.stay_at_home, mov_chance_stay_home, args.seed)
    create_environment(75) #size has been fixed at 75, but argument is left in case later versions add size control
    set_random(args.random, args.seed, args.v)
    initialise_results(args.ndts,args.nag, args.ninfect, nimmuno, nfree, nsocial)
    all_ages = age_dist(args.nag) 
    agents = create_agents(args.nag, all_ages, args.ninfect, nimmuno, nfree, nsocial)
    

    
    
    day = 0
    panic_factor = 0 #number represents size of square to look for areas with lower concentracion of people
    quarantine_policy = 0 #determines if infected people are quarantined (1 if activated)
    stay_home = 0 #stay at home order, if put at 1, the order is in place
    quarantined_agents = [] #list of quarantined agents (infected), initially empty
    
    for i in range(args.ndts):
        
    
        move(agents, panic_factor, i, day, stay_home,mov_chance_stay_home)
        infect(agents, i, day)
        death(agents, quarantined_agents, i, day)
        if (quarantine_policy):
            quarantine(agents, quarantined_agents, i, day )
        panic_factor, quarantine_policy, stay_home = update(day, args.day_measure, args.social_distancing, args.stay_at_home)
        
        if (args.visualize): #plot population per grid
            plt.figure(1)
            plt.imshow(ENV_DATA["population_density"])
            cbar = plt.colorbar()
            cbar.set_label('Agents per grid cell')
            plt.title("Population density per grid cell")
            plt.draw()
            plt.pause(0.001)
            plt.clf()
            

            
            
            
        IT_STATS["dt_of_the_day"] += 1
        if ((IT_STATS["dt_of_the_day"] % 17) == 0):
            day += 1
            IT_STATS["dt_of_the_day"] = 1
            
            if (args.visualize_infected): #update the plot with # of infected agents
                plt.figure(2)
                ax = plt.axes(xlim=(0, args.ndts/16), ylim=(0, args.nag))
                plt.title("Number of infected individuals per day")
                plt.ylabel('# of infected agents')
                plt.xlabel('Day')
                plt.plot(IT_STATS["total_infected_per_day"][0:day])
                plt.draw()
                plt.pause(0.001)
                plt.clf()
            
        progressbar(i+1, args.ndts, 50, i+1, day)
    
    
    
    
        
    
    
    plt.figure(2)
    plt.bar(range(len(IT_STATS["infections_per_day"])), IT_STATS["infections_per_day"], label = 'Infections per day'   )
    plt.title("infections per day")
    
    
    plt.figure(3)
    plt.plot(np.cumsum(IT_STATS["infections_per_day"]))
    plt.title("cumulative infections per day ")
    
    
    plt.figure(4)
    plt.plot(IT_STATS["total_infected_per_day"])
    plt.title("total_infected_per_day")
    
    
    plt.figure(5)
    plt.plot(IT_STATS["total_susceptible_per_day"])
    plt.title("total_susceptible_per_day")
    
    
    plt.figure(6)
    plt.bar(range(len(IT_STATS["deaths_per_day"])), IT_STATS["deaths_per_day"], label = 'Deaths per day'   )
    plt.title("deaths per day")   
    
    
    plt.figure(7)
    plt.plot(np.cumsum(IT_STATS["deaths_per_day"]))
    plt.title("cumulative deaths per day")
    
    
    plt.figure(8)
    plt.plot(np.cumsum(IT_STATS["recovered_per_day"]))
    plt.title("cumulative recoveries per day")
    
    
     
    
    
    if (args.foldersave == 'No folder input'):
        plt.show()
    else:
        print("saved results in folder: {}".format(args.foldersave))
           
    
    
    ##█SAVE THE RESULTS
    pwd = os.getcwd()
    if (args.foldersave == 'No folder input'):
        foldername = input("Please specify name of folder where results will be saved: ")
    else:
        foldername = args.foldersave
    
    savedir = os.path.join(pwd, foldername)
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    
    #changing to the directory where we will save our results
    os.chdir(savedir)
    
    with open('itstats.pickle', 'wb') as handle:
        pickle.dump(IT_STATS, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('paramdata.pickle', 'wb') as handle:
        pickle.dump(PARAMS, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    os.chdir('..') #go back to prev directory so that the function can be called again

        
        
if __name__ == '__main__':
    main()