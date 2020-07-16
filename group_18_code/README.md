# COVID.PY

covid.Py is a SIR model developed by Jorge Díez García-Victoria, Lana Abou Rafeh & Jignze Liu as the group project of COM6009: Modeling and Simulation of Natural Systems at the University of Sheffield, 2020.

Its main purpose is the simulation of the spread of COVID-19 in a 1:18 scale of the city of Sheffield.

## PACKAGES USED

covid.Py uses Human objects. The Human class and its corresponding functions, getters and setters are defined in Human.Py

Overall, both files use the following packages:

- pickle
- matpotlib
- numpy.ma
- numpy.matlib
- math (sin, cos, pi and ceil)
- scipy.spatial.distance
- os
- sys
- argparse
- pandas


covid.py also uses two separate excel spreadsheets. They need to be in the same folder as covid.py and human.py.

- The spreadsheet deathsbyagegroup contains the probability of death by cov-19 per age group
- The spreadsheet population_estimate_2018_ONS contains the population estimate of the UK in 2018

Both files are used to obtain the chances of death per timestep, and the age distribution of the agents in the simulation, respectively.



## DESCRIPTION AND USAGE

covid.py uses argparse, so the order of the arguments is irrelevant and all parameters have a default value.

covid.py is called by command with: Python covid.py

All arguments are therefore optional, so if the user wants to change the specific value of a parameter, he can do so by doing -parameter_in_question value. For example, If I wanted all parameters to remain default, but I wanted to have 10000 agents in my simulation I would need to call my program like this: Python covid.py -nag 10000

For a full list of all the parameters used in the simulation, their usage, and their corresponding description, the user can call the help function of covid.py by typing: python covid.py -h

It is worth mentioning that the visualize and visualize_infected parameters are especially useful if the user wants immediate visual output, but they slow the execution of the code considerably.

covid.py has a very simple progressbar that informs the user of how many timesteps have been executed, how many days have passed in the simulation and the percentage of the experiment that has been completed along with a text-based progress bar.

The code is thoroughly commented, so no further explanation on the code will be made in the readme. If there are any additional inquiries about the functioning of specific mechanisms / program logic, please use the following email:  jdiezgarcia-victoria1@sheffield.ac.uk

It is worth mentioning that even though the program supports the quarantining of infected agents, the functioning of this method and the interplay with the other methods was only tested, not implemented in experiments. A future direction of this SIR model would be to find an adequate policy for the quarantine of agents, where a percentage of agents should not never be quarantined, to account for asymptomatic cases.


## RESULTS OF A SIMULATION AND PICKLE

When a simulation ends, the program shows the most important statistics in plots, such as the cumulative infections per day, total amount of infected agents per day, cumulative deaths per day, etc.

When all the corresponding plots are closed, the user can then add a folder (if he hasn’t specified it directly by using the argument foldersave) by typing it into console. covid.py will then use pickle to serialize the dictionaries containing the parameters of interest and the corresponding results, which can later be called and plotted again with load_and_plot.py, explained later


## PARALLELIZED EXPERIMENTS 

Experiments do not take long to complete, but the combination of a specific set of parameters can inrease the duration of experiments up to 22 minutes. Due to the stochasticity of the simulator, one trial is not enough to obtain statistical significant results, so at least 10 trials are reccomended, with 100 being the optimal number of runs. 

Since this would lead to tens of hours of simulation time, Parallelization was introduced only to run many experiments, where computaiton time is reduced by almost 80%. 

The code experiment_selector.py imports covid.py and calls its main function asynchronously. The list of random seeds "seeds" was obainted randomly. The first 100 seeds were used for the results (p-values as well), and as such, the seeds have been left. 

## ADDITIONAL CODE FOR RESULT ANALYSIS

load_and_plot.py lets the user, via command console, create graphs to study the results of simulations obtained with covid.py. It is also capable of generating frames, to create short movies or gifs later on. 

The user can enter how many results he wants to plot, and then input the folders where the pickled results are located. The code then creates a plot with the number of infected agents per day. The user can also indicate if he wishes to obtain the corresponding frames. If this option is indicated, the code will save (in the specified folder) the png results per day. 