import ABC
import Simulation as Simulation
import numpy as np

# import matplotlib.pyplot as plt
# import Simulation as Simulation
# import time
# from Cost import objective as lqr_cost

# Set the iterations and the populations
iter = 10
pops = 10
max_iter = iter
n_employed_bees = pops
n_scout_bees = pops
n_onlooker_bees = pops

# Set the upper bound, the lower bound, and the limit for scout phase of the ABC
ub_th = 200
ub_phi = 20
lb_th = 0
lb_phi = 0
scout_trials_limit = 5
ub = np.array([ub_th, ub_phi])
lb = np.array([lb_th, lb_phi])

# Set the time sampling interval and the max time of the simulation
dt = .05  # set it to .05 for real-life time sampling interval
time_max = 5

# ===ABC-based optimization===
abc = ABC.ABC(n_employed_bees, n_scout_bees, n_onlooker_bees
              , lb, ub, scout_trials_limit, max_iter, dt, time_max)
solutions_list, costs_list, costs_of_each_iter = abc.optimize()

# ===Simulate the obtained solution of PID param===
sim = Simulation.simulation(dt, time_max)
sim.simulate_the_solution(solutions_list, costs_list, costs_of_each_iter)