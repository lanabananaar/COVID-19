#!/usr/bin/python

import os

import pickle
from matplotlib import pyplot as plt
import numpy as np
import numpy.matlib
from matplotlib.animation import FuncAnimation


"""
this function loads the results fr1om the inputted folder and plots the corresponding results
"""


nr_exp = int(input("Please write how many results you are going to plot: "))

experiments_folder_names = []
experiment_names = []
for i in range(nr_exp):
    folder_name = input("please write name of experiment folder in order: ")
    experiments_folder_names.append(folder_name) 
    experiment_name = input("please write the name of the experiment: ")
    experiment_names.append(experiment_name)


pwd = os.getcwd()
ALL_IT_STATS = []
ALL_PARAMS = []

for folder in experiments_folder_names:
    
    resultsdir = os.path.join(pwd, folder)

    if not os.path.exists(resultsdir):
        exit("the folder does not exist, please check again")

    #changing to the directory where we will get our results
    os.chdir(resultsdir)
    
    #load the results
    with open('itstats.pickle', 'rb') as handle:
        ALL_IT_STATS.append(pickle.load(handle))
        
    with open('paramdata.pickle', 'rb') as handle:
        ALL_PARAMS.append(pickle.load(handle))
    
    
 
for i in range(nr_exp):
    plt.figure(1)
    plt.plot(ALL_IT_STATS[i]["total_infected_per_day"], label = experiment_names[i] )
    plt.title("Number of infected individuals of a initial population of 5000")
    plt.ylabel('# of infected individuals')
    plt.xlabel('Day')
    plt.legend(experiment_names)


plt.show()








frame_checker = input ("Write name of folder to save frames or write NO:  ")


if (frame_checker != "NO"):
    resultsdir = os.path.join(pwd, frame_checker)
    example_data = ALL_IT_STATS[0]["total_infected_per_day"]
    
    fig = plt.figure(figsize=(13.5,8))
    
    

    
    for j in range(example_data.size):
        ax = plt.axes(xlim=(0, example_data.size), ylim=(example_data.min(), example_data.max()))
        plt.title("# Of infected individuals of a initial population of 5000")
        plt.ylabel('# of infected individuals')
        plt.xlabel('Day')
        for exp in range(nr_exp):
            plt.plot(ALL_IT_STATS[exp]["total_infected_per_day"][0:j+1], label = experiment_names[exp] )
            plt.legend(experiment_names)
        plt.draw()
        plt.pause(0.001)
        fig.savefig(resultsdir + "\\"  + 'frame' + str(j+1) + '.png')
        plt.clf()

    
    
    
    
    
    
    
    
"""    
plt.figure(1)
plt.bar(range(len(IT_STATS["infections_per_day"])), IT_STATS["infections_per_day"], label = 'Infections per day'   )
plt.title("infections per day")

plt.figure(2)
plt.plot(np.cumsum(IT_STATS["infections_per_day"]))
plt.title("cumulative infections per day ")
#plt.show()



plt.figure(3)
plt.plot(IT_STATS["total_infected_per_day"])
plt.title("total_infected_per_day")
#plt.show()




plt.figure(4)
plt.plot(IT_STATS["total_susceptible_per_day"])
plt.title("total_susceptible_per_day")
#plt.show()




plt.figure(5)
plt.bar(range(len(IT_STATS["deaths_per_day"])), IT_STATS["deaths_per_day"], label = 'Deaths per day'   )
plt.title("deaths per day")   





plt.figure(6)
plt.plot(np.cumsum(IT_STATS["deaths_per_day"]))
plt.title("cumulative deaths per day")
#plt.show()



plt.figure(7)
plt.plot(np.cumsum(IT_STATS["recovered_per_day"]))
plt.title("cumulative recoveries per day")
#plt.show()




plt.figure(8)
plt.plot(IT_STATS["quarantined_per_day"])
plt.title("quarantines per day")
#plt.show()


plt.figure(9)
plt.plot(IT_STATS["total_quarantined_per_day"])
plt.title("total_quarantined_per_day per day")
#plt.show()


plt.show()
"""    
    
  
    
os.chdir('..') #go back to prev directory so that the function can be called again
